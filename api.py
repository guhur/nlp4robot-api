#!/usr/bin/env python3
"""
Run an RESTful API with Flask for getting samples from nalanbot
"""
import os
from typing import Dict, Any
from copy import deepcopy
from flask import Flask, request
from flask_restful import Resource, Api, abort, reqparse, fields, marshal_with
from flask_cors import CORS
import nalanbot as nb
from nalanbot.experiments import ExperimentManager
from nalanbot.training import supervised_init, student_forcing, teacher_forcing, supervised_default_params
from nalanbot.sims import act_on_states, BulletBody, BulletManipulator
from nalanbot.visualizations import JSONCollector
from nalanbot.metrics import is_tower_built
from nalanbot.abstracts import SeqSample, Sample
from nalanbot.tokenizers import SampleTokenizer, ActionTokenizer

# As of v0.11, it's the AMT testset
DEFAULT_DATASET = "dataset-21"

class Context:

    def __init__(self):
        self.manager: nb.ExperimentManager = None
        self.dataset: nb.BaseDataset = None
        self.dataset_name: str = DEFAULT_DATASET
        self.policy: nb.BasePolicy = None
        self.sim: nb.BaseSim = None
        self.renderer: nb.BaseRender = None
        self.sample_tokenizer: nb.SampleTokenizer = None
        self.action_tokenizer: nb.ActionTokenizer = None
        self.collector: JSONCollector = None
        self.context: Dict[str, Any] = {}

    def init(self, **context: Any) -> None:
        if context == self.context:
            return

        if "dataset_name" in context:
            self.dataset_name = context["dataset_name"]

        # Update manager with context
        params = {
                "testsets": {},
                "num_workers": 0,
                **context,
                }
        self.manager = ExperimentManager()
        supervised_default_params(self.manager)
        self.collector = JSONCollector(self.manager)

        # Load
        self.policy, opt, datagens, self.sim, self.renderer = supervised_init(self.manager)
        self.policy, _, _ = self.manager.load(self.policy, opt)

        dataset_path = nb.get_config("datasets") / self.dataset_name
        self.dataset = nb.DeterministicLMDBDataset(folder=dataset_path)
        self.sample_tokenizer = SampleTokenizer(
            device=self.manager.get("device"), workspace=self.sim.workspace
        )
        self.action_tokenizer = ActionTokenizer(workspace=self.sim.workspace)

    def sampling(self) -> Sample:
        sample = next(self.dataset)
        return sample

    def _teacher_forcing(self, samples: SeqSample) :
        _, tactions, _ = teacher_forcing(
            self.policy, samples, self.sim, self.renderer, self.sample_tokenizer, self.action_tokenizer
        )
        tf_actions = self.action_tokenizer.decode(tactions)
        tf_states = act_on_states(self.sim, tf_actions, deepcopy(samples.init_state))
        tf_result = is_tower_built(samples.batch[0], tf_actions[0], tf_states[0])
        return tf_actions[0], tf_states[0], tf_result

    def _student_forcing(self, samples: SeqSample):
        tsamples, tactions, _ = student_forcing(
            self.policy, samples, self.sim, self.renderer, self.sample_tokenizer, self.action_tokenizer
        )
        sf_states = deepcopy(tsamples.states)
        sf_actions = self.action_tokenizer.decode(tactions)
        sf_result = is_tower_built(samples.batch[0], sf_actions[0], sf_states[0])
        return sf_actions[0], sf_states[0], sf_result

    def predict(self, samples: SeqSample) -> str:
        tf_action, tf_state, tf_result = self._teacher_forcing(samples)
        sf_action, sf_state, sf_result = self._student_forcing(samples)

        self.collector.add_sample(
            samples.batch[0],
            {
                "mode": "student",
                "actions": sf_action,
                "states": sf_state,
                "result": sf_result,
            },
            {
                "mode": "teacher",
                "actions": tf_action,
                "states": tf_state,
                "result": tf_result,
            },
        )
        return self.collector.export()



body_fields = {
        "position": fields.List(fields.Float),
        "size": fields.List(fields.Float),
        "shape": fields.String,
        "color_name": fields.String,
        "color": fields.List(fields.Float),
        }

manipulator_fields = {
        "position": fields.List(fields.Float),
        "orientation": fields.List(fields.Float),
        }

state_fields = {
        "bodies": fields.List(fields.Nested(body_fields)),
        "manipulator": fields.Nested(manipulator_fields)
        }

sample_fields = {
        "sentence": fields.String,
        "action": fields.List(fields.String),
        "color": fields.List(fields.String),
        "position": fields.List(fields.List(fields.Float)),
        "states": fields.List(fields.Nested(state_fields))
}

class AskSample(Resource):

    def __init__(self, context: Context):
        self.context = context

    @marshal_with(sample_fields, envelope='resource')
    def post(self):
        json_data = request.get_json(force=True)

        if "context" not in json_data:
            abort(400, message="Missing the context")

        self.context.init(**json_data['context'])
        sample: Sample = self.context.sampling()

        return {
            "sample": sample,
            "dataset_name": self.context.dataset_name
        }


class PredictFromSample(Resource):

    def __init__(self, context: Context):
        self.context = context

    def post(self):
        json_data = request.get_json(force=True)

        if "context" not in json_data:
            abort(400, message="Missing the context")
        if "sample" not in json_data:
            abort(400, message="Missing the sample")

        self.context.init(**json_data['context'])

        # FIXME dirty code
        # import pdb; pdb.set_trace()
        json_states: Dict = json_data['sample']['states']
        states = []
        for state in json_states:
            manipulator = BulletManipulator(**state['manipulator'])
            bodies = [BulletBody(**json_body) for json_body in state['bodies']]
            states.append({
                "manipulator": manipulator,
                "bodies": bodies
                })
        del json_data['sample']['states']

        sample: Sample = Sample(**json_data['sample'], states=states)
        seq_sample: SeqSample = SeqSample.from_batch([sample])
        data: str = self.context.predict(seq_sample)
        return data


def create_app(prefix: str = "/", debug: bool = False) -> Flask:
    app = Flask(__name__)

    if debug:
        cors = CORS(app, resources={r"*": {"origins": "*"}})

    context = Context()

    api = Api(app)
    api.add_resource(AskSample, prefix + "/sample/", resource_class_args=(context, ))
    api.add_resource(PredictFromSample, prefix + "/predict/", resource_class_args=(context, ))

    app.run(debug=debug)
    return app


if __name__ == '__main__':
    debug: bool = bool(os.environ.get("API_DEBUG", True))
    prefix: str = os.environ.get("API_PREFIX", "/api")

    app: Flask = create_app(prefix, debug)
