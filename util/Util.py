# -*- coding: utf-8


class Util:

    @staticmethod
    def extract_train_and_test_data(data):
        train_data = data['train']
        train_x = train_data['x']
        train_y = train_data['y']
        test_data = data['test']
        test_x = test_data['x']
        test_y = test_data['y']
        return [train_x, train_y], [test_x, test_y]
