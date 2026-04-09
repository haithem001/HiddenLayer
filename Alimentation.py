class Alimentation :

    #Energy     :Carbohydrates ,Fat
    #Functional :Proteins
    #Regulators :Vitamins ,Minerals
    #Hydration  :Water and stuff

    def __init__(self ,energy,functional,regulator,hydration):
        self.energy=energy
        self.functional=functional
        self.regulator=regulator
        self.hydration=hydration

    def set_energy(self,energy):
        self.energy=energy
    def set_functional(self,functional):
        self.functional=functional
    def set_regulator(self,regulator):
        self.regulator=regulator
    def set_hydration(self,hydration):
        self.hydration=hydration


    def get_energy(self):
        return self.energy
    def get_functional(self):
        return self.functional
    def get_regulator(self):
        return self.regulator
    def get_hydration(self):
        return self.hydration

class Air:
    def __init__(self, oxygen, carbon_dioxide, nitrogen):

        self.oxygen = oxygen
        self.carbon_dioxide = carbon_dioxide
        self.nitrogen = nitrogen


class Respiration:
    def __init__(self, air: Air):
        self.air = air

    def inspiration(self):

        return self.air

    def gas_exchange(self):

        self.air.oxygen -= 5
        self.air.carbon_dioxide += 4

    def expiration(self):

        return self.air





