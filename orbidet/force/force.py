from .acceleration import Acceleration

class Force():

    def __init__(self, integrationFrame = "TOD", gravityFrame = "PEF"):
        self._force = []
        self.integrationFrame = integrationFrame
        self.gravityFrame = gravityFrame


    def addForce(self, force):
        assert isinstance(force, Acceleration), "Input force variable should be of type Orbidet.force.force.Acceleration"
        self._force.append(force)

    def removeForce(self,force):
        assert isinstance(force, Acceleration), "Input force variable should be of type Orbidet.force.force.Acceleration"
        self._force.remove(force)

    def __repr__(self):
        str = "Active Forces: "

        for force in self._force:
            str += force.name + ", "
        return str[0:-2]

    def getForceList(self):
        return self._force


    @property
    def tesserals(self):
        if not hasattr(self, "_tesserals"):
            for force in self._force:
                if "gravity" in force.type:
                    self.tesserals = force.order >= 1
        return self._tesserals

    @tesserals.setter
    def tesserals(self, val):
        self._tesserals = val
