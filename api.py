#!/usr/bin/env python3
"""
Run an RESTful API with Flask for getting samples from nalanbot
"""
import os
from typing import Dict, Any
from copy import deepcopy
from flask import Flask, request
from flask_restful import Resource, Api, abort, reqparse
from nalanbot.experiments import ExperimentManager
from nalanbot.training import supervised_init, student_forcing, teacher_forcing
from nalanbot.sims import act_on_states
from nalanbot.visualizations import JSONCollector
from nalanbot.metrics import is_tower_built
from nalanbot.abstracts import SeqSample, Sample
from nalanbot.tokenizers import SampleTokenizer, ActionTokenizer


class Context:

    def __init__(self):
        self.manager = None
        self.dataset = None
        self.policy = None
        self.sim = None
        self.renderer = None
        self.sample_tokenizer = None
        self.action_tokenizer = None
        self.collector = None
        self.context = {}

    def init(self, **context: Any) -> None:
        if context == self.context:
            return

        # Update manager with context
        self.manager = ExperimentManager(**context)
        self.collector = JSONCollector(self.manager)

        # Load
        self.policy, opt, datagens, self.sim, self.renderer = supervised_init(self.manager)
        self.policy, _ = self.manager.load(self.policy, opt)
        self.dataset = datagens[1].dataset
        self.sample_tokenizer = SampleTokenizer(
            device=self.manager.get("device"), workspace=self.sim.workspace
        )
        self.action_tokenizer = ActionTokenizer(workspace=self.sim.workspace)

    def sampling(self) -> SeqSample:
        sample = next(self.dataset)
        samples = SeqSample.from_batch([sample])
        return samples

    def _teacher_forcing(self, samples: SeqSample) -> Dict:
        _, tactions, _ = teacher_forcing(
            self.policy, samples, self.sim, self.renderer, self.sample_tokenizer, self.action_tokenizer
        )
        tf_actions = self.action_tokenizer.decode(tactions)
        tf_states = act_on_states(self.sim, tf_actions, deepcopy(samples.init_state))
        tf_result = is_tower_built(samples.batch[0], tf_actions[0], tf_states[0])
        return tf_actions[0], tf_states[0], tf_result

    def _student_forcing(self, samples: SeqSample) -> Dict:
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



class AskSample(Resource):

    def __init__(self, context: Context):
        self.context = context

    def post(self):
        json_data = request.get_json(force=True)
        self.context.init(**json_data['context'])
        samples = self.context.sampling()
        return self.context.sampling(samples)


class PredictFromSample(Resource):

    def __init__(self, context: Context):
        self.context = context

    def post(self):
        json_data = request.get_json(force=True)
        self.context.init(**json_data['context'])
        sample: Sample = Sample(**json_data['sample'])
        seq_sample: SeqSample = SeqSample.from_sample(sample)
        return self.context.predict(seq_sample)

def create_app(prefix="/", debug=False):
    app = Flask(__name__)
    api = Api(app)
    context = Context()

    api.add_resource(AskSample, prefix + "/sample", resource_class_args=(context, ))
    api.add_resource(PredictFromSample, prefix + "/predict", resource_class_args=(context, ))
    app.run(debug=debug)
    return app


if __name__ == '__main__':
    debug = os.environ.get("API_DEBUG", True)
    prefix = os.environ.get("API_PREFIX", "/api")

    app = create_app(prefix, debug)
