class Model:
    ticks = 0

    def __init__(self, data, filename, num_tests=10, test_size=10):
        self.data = data
        self.filename = filename
        self.num_tests = num_tests
        self.test_size = test_size
        self.errors = []

    def __repr__(self):
        return self.filename.split(".")[0]

    @classmethod
    def increment_ticker(cls):
        cls.ticks += 1

