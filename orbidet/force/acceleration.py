class Acceleration():

    def __init__(self,name):
        self.name = name
        print(name)
        # facilitate access to each acceleration type (not very elegant)
        if "Drag" in name:
            self.type = "drag"
        elif "Gravity" in name:
            self.type = "gravity"
        elif "Two Body" in name:
            self.type = "central"


    def acceleration(self,orbit):
        pass

    def getName(self):
        return self.name
