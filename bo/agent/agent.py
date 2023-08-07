
class Agent():
    def __init__(self, model, x_train, y_train, region_support) -> None:
        self.model = model
        self.point_history = []
        self.x_train = x_train
        self.y_train = y_train
        self.region_support = region_support
        
    def add_point(self, point):
        self.point_history.append(point)

    def update_model(self, model):
        self.model = model

    def update_bounds(self, region_support):
        self.region_support = region_support

