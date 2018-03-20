
class Tester:
    TEST_MODE_GENERATIVE = 'generative mode'
    TEST_MODE_PREDICTION = 'prediction mode'

    def __init__(self, test_mode=TEST_MODE_GENERATIVE):
        self.test_mode = test_mode

    def test(self, esn, data):
        pass

