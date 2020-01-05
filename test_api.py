import unittest
import os
import json
from api import create_app


class APITestCase(unittest.TestCase):
    """This class represents the API test case"""

    def setUp(self):
        """Define test variables and initialize app."""
        self.app = create_app(debug=True, prefix="/api/")
        self.client = self.app.test_client

    def test_generate_sample(self):
        """Test API can generate a sample and its prediction (POST request)"""
        context = {"name": "test",
                "policy_name": "film"
                }
        res = self.client().post('/api/', data=context)
        self.assertEqual(res.status_code, 201)
        print(res.data)
        self.assertIn('Go to Borabora', str(res.data))


# Make the tests conveniently executable
if __name__ == "__main__":
    unittest.main()
