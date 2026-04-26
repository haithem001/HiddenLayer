class Object3D:
    def __init__(self , Coordinates , Total_mass, Light, components):
        self.x = Coordinates[0]
        self.y = Coordinates[1]
        self.z = Coordinates[2]
        self.Total_mass = Total_mass
        self.components = components







        pass

class Component3D(Object3D):
    def __init__(self, Coordinates, Total_mass, Light, components):
        super().__init__(self, Coordinates, Total_mass, Light, components)




