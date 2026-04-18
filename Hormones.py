import math

SETPOINTS = {
    # Group 1
    "Dopamine":        0.50,
    "Serotonin":       0.50,
    "Noradrenaline":   0.30,
    "Adrenaline":      0.10,
    "Cortisol":        0.20,
    "Oxytocin":        0.30,
    "Endorphin":       0.30,
    "Melatonin":       0.10,
    "Cortistatin":     0.20,
    # Group 2
    "Ghrelin":         0.30,
    "Insulin":         0.50,
    "Glucagon":        0.20,
    "Cholecystokinin": 0.20,
    "Leptin":          0.50,
    "Testosterone":    0.40,
    "Estrogen":        0.40,
    "Vasopressin":     0.30,
    "Aldosterone":     0.30,
    # Group 3
    "T3":              0.50,
    "Erythropoietin":  0.30,
    "ANP":             0.20,
    "Prostaglandins":  0.05,
    "GH":              0.30,
}

# ============================================================
# CONSTANTS
# alpha  = production/stimulation rate
# beta   = natural decay rate
# gamma  = inhibition coefficient
# delta  = event spike amplitude
# epsilon= secondary decay
# zeta   = event suppression amplitude
# homeo  = homeostatic pull strength (NEW)
# self_i = self-inhibition to prevent runaway (NEW)
# ============================================================
CONSTANTS = {
    # Group 1 — slowed down significantly, added homeostasis
    "Dopamine": {
        "alpha": 0.03, "beta": 0.01, "gamma": 0.05,
        "delta": 0.20, "homeo": 0.008, "self_i": 0.0
    },
    "Serotonin": {
        "alpha": 0.02, "beta": 0.005, "gamma": 0.04,
        "delta": 0.03, "epsilon": 0.02, "homeo": 0.006
    },
    "Noradrenaline": {
        "alpha": 0.05, "beta": 0.02, "gamma": 0.03,
        "delta": 0.15, "epsilon": 0.05, "homeo": 0.008
    },
    "Adrenaline": {
        "alpha": 0.20, "beta": 0.08, "gamma": 0.0,
        "delta": 0.05, "homeo": 0.01
    },
    "Cortisol": {
        "alpha": 0.008, "beta": 0.005, "gamma": 0.03,
        "delta": 0.02,  "epsilon": 0.0, "zeta": 0.05,
        "homeo": 0.005, "self_i": 0.02   # self-inhibits at high levels
    },
    "Oxytocin": {
        "alpha": 0.15, "beta": 0.01, "gamma": 0.04,
        "delta": 0.02, "homeo": 0.006
    },
    "Endorphin": {
        "alpha": 0.05, "beta": 0.02, "gamma": 0.0,
        "delta": 0.10, "homeo": 0.005
    },
    "Melatonin": {
        "alpha": 0.10, "beta": 0.02, "gamma": 0.08,
        "homeo": 0.004
    },
    "Cortistatin": {
        "alpha": 0.05, "beta": 0.01, "gamma": 0.02,
        "homeo": 0.004
    },

    # Group 2
    "Ghrelin": {
        "alpha": 0.005, "beta": 0.0, "gamma": 0.02,
        "delta": 0.10,  "homeo": 0.003
    },
    "Insulin": {
        "alpha": 0.15, "beta": 0.008, "gamma": 0.02,
        "delta": 0.05, "homeo": 0.005
    },
    "Glucagon": {
        "alpha": 0.06, "beta": 0.01, "gamma": 0.0,
        "delta": 0.08, "homeo": 0.005
    },
    "Cholecystokinin": {
        "alpha": 0.20, "beta": 0.05,
        "homeo": 0.005
    },
    "Leptin": {
        "alpha": 0.002, "beta": 0.001,
        "homeo": 0.001   # very slow — long term signal
    },
    "Testosterone": {
        "alpha": 0.05, "beta": 0.005, "gamma": 0.03,
        "delta": 0.02, "homeo": 0.004
    },
    "Estrogen": {
        "alpha": 0.04, "beta": 0.004, "gamma": 0.0,
        "delta": 0.02, "homeo": 0.004
    },
    "Vasopressin": {
        "alpha": 0.08, "beta": 0.02,
        "homeo": 0.005
    },
    "Aldosterone": {
        "alpha": 0.04, "beta": 0.01,
        "homeo": 0.004
    },

    # Group 3
    "T3": {
        "alpha": 0.01, "beta": 0.004, "gamma": 0.02,
        "homeo": 0.003
    },
    "Erythropoietin": {
        "alpha": 0.06, "beta": 0.008,
        "homeo": 0.004
    },
    "ANP": {
        "alpha": 0.08, "beta": 0.012,
        "homeo": 0.004
    },
    "Prostaglandins": {
        "alpha": 0.15, "beta": 0.04,
        "homeo": 0.003
    },
    "GH": {
        "alpha": 0.10, "beta": 0.015,
        "homeo": 0.004
    },
}


def clamp(value, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, value))


def homeo_pull(hormone_name, current_value):
    """
    Homeostatic pull: gentle force returning hormone to its setpoint.
    Acts like a spring — stronger when further from setpoint.
    """
    k = CONSTANTS[hormone_name]
    sp = SETPOINTS[hormone_name]
    return k.get("homeo", 0.005) * (sp - current_value)


def self_inhibit(hormone_name, current_value):
    """
    Self-inhibition: hormone suppresses its own production when too high.
    Prevents runaway like cortisol maxing out.
    """
    k = CONSTANTS[hormone_name]
    si = k.get("self_i", 0.0)
    if si == 0.0:
        return 0.0
    return si * current_value * current_value


# ============================================================
# GROUP 1 — Consciousness & Motivation
# ============================================================

class Group1:
    def __init__(self, Dopamine, Serotonin, Noradrenaline,
                 Adrenaline, Cortisol, Oxytocin, Endorphin,
                 Melatonin, Cortistatin):
        self.Dopamine      = Dopamine
        self.Serotonin     = Serotonin
        self.Noradrenaline = Noradrenaline
        self.Adrenaline    = Adrenaline
        self.Cortisol      = Cortisol
        self.Oxytocin      = Oxytocin
        self.Endorphin     = Endorphin
        self.Melatonin     = Melatonin
        self.Cortistatin   = Cortistatin

    def update(self, dt, env, events):
        c = CONSTANTS

        Da = self.Dopamine
        Se = self.Serotonin
        Na = self.Noradrenaline
        Ad = self.Adrenaline
        Co = self.Cortisol
        Ox = self.Oxytocin
        En = self.Endorphin
        Mt = self.Melatonin
        Cs = self.Cortistatin

        Al  = env.get("activity_level", 0.0)
        D   = env.get("darkness", 0.0)
        L   = env.get("light", 1.0)
        Pl  = env.get("pain_level", 0.0)
        sig = env.get("safety_level", 0.5)
        In  = env.get("insulin", 0.5)
        Es  = env.get("estrogen", 0.4)

        E_success = events.get("success", 0)
        E_failure = events.get("failure", 0)
        E_threat  = events.get("threat", 0)
        E_social  = events.get("social_success", 0)

        # DOPAMINE
        # production from oxytocin*noradrenaline interaction
        # inhibited by cortisol, boosted by success event
        # homeostasis pulls back to 0.5
        k = c["Dopamine"]
        dDa = (k["alpha"] * (Ox * Na)
               - k["beta"]  * Da
               - k["gamma"] * Co * Da
               + k["delta"] * E_success
               + homeo_pull("Dopamine", Da))
        self.Dopamine = clamp(Da + dDa * dt)

        # SEROTONIN
        # production from estrogen and dopamine, boosted by activity
        # inhibited by cortisol and darkness (converts to melatonin at night)
        k = c["Serotonin"]
        dSe = (k["alpha"]   * (Es + Da)
               + k["beta"]    * Al
               - k["gamma"]   * Co * Se
               - k["delta"]   * Se * D
               - k["epsilon"] * Se
               + homeo_pull("Serotonin", Se))
        self.Serotonin = clamp(Se + dSe * dt)

        # NORADRENALINE
        # driven by adrenaline and threat, suppressed by serotonin and melatonin
        k = c["Noradrenaline"]
        dNa = (k["alpha"]   * Ad
               + k["delta"]   * E_threat
               - k["gamma"]   * Se * Na
               - k["epsilon"] * Mt * Na
               - k["beta"]    * Na
               + homeo_pull("Noradrenaline", Na))
        self.Noradrenaline = clamp(Na + dNa * dt)

        # ADRENALINE
        # spikes on threat, driven by cortisol and low insulin
        # fast natural decay, homeostasis brings it back down quickly
        k = c["Adrenaline"]
        dAd = (k["alpha"] * E_threat
               + k["delta"] * Co
               + k["gamma"] * (1.0 - In)
               - k["beta"]  * Ad
               + homeo_pull("Adrenaline", Ad))
        self.Adrenaline = clamp(Ad + dAd * dt)

        # CORTISOL
        # rises on failure, driven by adrenaline and low dopamine
        # CRITICAL: self-inhibits at high levels (HPA axis negative feedback)
        # oxytocin suppresses it, success reduces it
        k = c["Cortisol"]
        dCo = (k["alpha"] * E_failure
               + k["beta"]  * Ad
               + k["gamma"] * (1.0 - Da)
               - k["delta"] * Ox * Co
               - k["zeta"]  * E_success
               - self_inhibit("Cortisol", Co)
               + homeo_pull("Cortisol", Co))
        self.Cortisol = clamp(Co + dCo * dt)

        # OXYTOCIN
        # social bonding hormone, safety driven
        # suppressed by cortisol
        k = c["Oxytocin"]
        dOx = (k["alpha"] * E_social
               + k["delta"] * sig
               - k["gamma"] * Co * Ox
               - k["beta"]  * Ox
               + homeo_pull("Oxytocin", Ox))
        self.Oxytocin = clamp(Ox + dOx * dt)

        # ENDORPHIN
        # physical effort and pain release endorphins
        # sustains C? loop by masking pain
        k = c["Endorphin"]
        dEn = (k["alpha"] * Al
               + k["delta"] * Pl
               - k["beta"]  * En
               + homeo_pull("Endorphin", En))
        self.Endorphin = clamp(En + dEn * dt)

        # MELATONIN
        # rises with darkness*serotonin, suppressed by light
        # high melatonin suppresses all C nodes
        k = c["Melatonin"]
        dMt = (k["alpha"] * D * Se
               - k["gamma"] * L * Mt
               - k["beta"]  * Mt
               + homeo_pull("Melatonin", Mt))
        self.Melatonin = clamp(Mt + dMt * dt)

        # CORTISTATIN
        # rises with melatonin, slows C? firing rate
        # suppressed by noradrenaline
        k = c["Cortistatin"]
        dCs = (k["alpha"] * Mt
               - k["gamma"] * Na * Cs
               - k["beta"]  * Cs
               + homeo_pull("Cortistatin", Cs))
        self.Cortistatin = clamp(Cs + dCs * dt)


# ============================================================
# GROUP 2 — Body & Necessity Triggers
# ============================================================

class Group2:
    def __init__(self, Ghrelin, Insulin, Glucagon,
                 Cholecystokinin, Leptin, Testosterone,
                 Estrogen, Vasopressin, Aldosterone):
        self.Ghrelin          = Ghrelin
        self.Insulin          = Insulin
        self.Glucagon         = Glucagon
        self.Cholecystokinin  = Cholecystokinin
        self.Leptin           = Leptin
        self.Testosterone     = Testosterone
        self.Estrogen         = Estrogen
        self.Vasopressin      = Vasopressin
        self.Aldosterone      = Aldosterone

    def update(self, dt, env, events):
        c = CONSTANTS

        Gh = self.Ghrelin
        In = self.Insulin
        Gc = self.Glucagon
        Ck = self.Cholecystokinin
        Lp = self.Leptin
        Ts = self.Testosterone
        Es = self.Estrogen
        Vp = self.Vasopressin
        Al = self.Aldosterone

        Fi  = env.get("food_intake", 0.0)
        Act = env.get("activity_level", 0.0)
        Bs  = env.get("blood_sugar", 0.5)
        Rf  = env.get("fat_reserves", 0.5)
        Hl  = env.get("hydration", 0.5)
        rho = env.get("blood_pressure", 0.5)
        sig = env.get("safety_level", 0.5)
        Da  = env.get("dopamine", 0.5)
        Co  = env.get("cortisol", 0.2)

        E_dominance = events.get("dominance", 0)
        E_social    = events.get("social_bond", 0)

        # GHRELIN — hunger
        # rises slowly every tick (natural hunger building)
        # suppressed by cholecystokinin (satiety) and leptin (fat reserves)
        # homeostasis prevents it going to zero permanently
        k = c["Ghrelin"]
        dGh = (k["alpha"]
               - k["delta"] * Ck
               - k["gamma"] * Lp * Gh
               + homeo_pull("Ghrelin", Gh))
        self.Ghrelin = clamp(Gh + dGh * dt)

        # INSULIN
        # rises after food, stimulated by glucagon signal
        # consumed by activity, natural decay
        k = c["Insulin"]
        dIn = (k["alpha"] * Fi
               + k["delta"] * Gc
               - k["gamma"] * Act * In
               - k["beta"]  * In
               + homeo_pull("Insulin", In))
        self.Insulin = clamp(In + dIn * dt)

        # GLUCAGON — emergency energy
        # inversely proportional to insulin and blood sugar
        k = c["Glucagon"]
        dGc = (k["alpha"] * (1.0 - In)
               + k["delta"] * (1.0 - Bs)
               - k["beta"]  * Gc
               + homeo_pull("Glucagon", Gc))
        self.Glucagon = clamp(Gc + dGc * dt)

        # CHOLECYSTOKININ — satiety
        # spikes after eating, fast decay
        k = c["Cholecystokinin"]
        dCk = (k["alpha"] * Fi
               - k["beta"]  * Ck
               + homeo_pull("Cholecystokinin", Ck))
        self.Cholecystokinin = clamp(Ck + dCk * dt)

        # LEPTIN — fat reserves signal
        # very slow dynamics — reflects long term energy stores
        k = c["Leptin"]
        dLp = (k["alpha"] * Rf
               - k["beta"]  * Lp
               + homeo_pull("Leptin", Lp))
        self.Leptin = clamp(Lp + dLp * dt)

        # TESTOSTERONE
        # stimulated by dominance events and dopamine
        # destroyed by chronic high cortisol
        k = c["Testosterone"]
        dTs = (k["alpha"] * E_dominance
               + k["delta"] * Da
               - k["gamma"] * Co * Ts
               - k["beta"]  * Ts
               + homeo_pull("Testosterone", Ts))
        self.Testosterone = clamp(Ts + dTs * dt)

        # ESTROGEN
        # social bonding and safety driven
        # directly upregulates serotonin synthesis
        k = c["Estrogen"]
        dEs = (k["alpha"] * E_social
               + k["delta"] * sig
               - k["beta"]  * Es
               + homeo_pull("Estrogen", Es))
        self.Estrogen = clamp(Es + dEs * dt)

        # VASOPRESSIN — thirst
        # rises when dehydrated, homeostasis at low resting level
        k = c["Vasopressin"]
        dVp = (k["alpha"] * (1.0 - Hl)
               - k["beta"]  * Vp
               + homeo_pull("Vasopressin", Vp))
        self.Vasopressin = clamp(Vp + dVp * dt)

        # ALDOSTERONE — blood pressure / salt balance
        k = c["Aldosterone"]
        dAl = (k["alpha"] * (1.0 - rho)
               - k["beta"]  * Al
               + homeo_pull("Aldosterone", Al))
        self.Aldosterone = clamp(Al + dAl * dt)


# ============================================================
# GROUP 3 — Autonomous Biological Processes
# ============================================================

class Group3:
    def __init__(self, T3, Erythropoietin,
                 ANP, Prostaglandins, GH):
        self.T3              = T3
        self.Erythropoietin  = Erythropoietin
        self.ANP             = ANP
        self.Prostaglandins  = Prostaglandins
        self.GH              = GH

    def update(self, dt, env):
        c = CONSTANTS

        T3 = self.T3
        Ep = self.Erythropoietin
        Ap = self.ANP
        Pg = self.Prostaglandins
        Gh = self.GH

        Lp  = env.get("leptin", 0.5)
        O2  = env.get("oxygen_level", 0.8)
        rho = env.get("blood_pressure", 0.5)
        Dmg = env.get("damage_level", 0.0)
        Mt  = env.get("melatonin", 0.1)
        psi = env.get("sleep_depth", 0.0)

        # T3 — metabolic rate
        # slows under starvation (low leptin)
        k = c["T3"]
        dT3 = (k["alpha"]
               - k["gamma"] * (1.0 - Lp) * T3
               - k["beta"]  * T3
               + homeo_pull("T3", T3))
        self.T3 = clamp(T3 + dT3 * dt)

        # ERYTHROPOIETIN — oxygen response
        k = c["Erythropoietin"]
        dEp = (k["alpha"] * (1.0 - O2)
               - k["beta"]  * Ep
               + homeo_pull("Erythropoietin", Ep))
        self.Erythropoietin = clamp(Ep + dEp * dt)

        # ANP — heart pressure
        # only activates above baseline pressure
        k = c["ANP"]
        dAp = (k["alpha"] * max(0.0, rho - 0.5)
               - k["beta"]  * Ap
               + homeo_pull("ANP", Ap))
        self.ANP = clamp(Ap + dAp * dt)

        # PROSTAGLANDINS — pain and inflammation
        k = c["Prostaglandins"]
        dPg = (k["alpha"] * Dmg
               - k["beta"]  * Pg
               + homeo_pull("Prostaglandins", Pg))
        self.Prostaglandins = clamp(Pg + dPg * dt)

        # GROWTH HORMONE — repair during deep sleep
        k = c["GH"]
        dGh = (k["alpha"] * (Mt * psi)
               - k["beta"]  * Gh
               + homeo_pull("GH", Gh))
        self.GH = clamp(Gh + dGh * dt)


# ============================================================
# INTERNAL STATE VECTOR
# ============================================================

class InternalStateVector:
    def __init__(self):
        self.G1 = Group1(
            Dopamine=0.50, Serotonin=0.50, Noradrenaline=0.30,
            Adrenaline=0.10, Cortisol=0.20, Oxytocin=0.30,
            Endorphin=0.30, Melatonin=0.10, Cortistatin=0.20
        )
        self.G2 = Group2(
            Ghrelin=0.30, Insulin=0.50, Glucagon=0.20,
            Cholecystokinin=0.20, Leptin=0.50, Testosterone=0.40,
            Estrogen=0.40, Vasopressin=0.30, Aldosterone=0.30
        )
        self.G3 = Group3(
            T3=0.50, Erythropoietin=0.30,
            ANP=0.20, Prostaglandins=0.05, GH=0.30
        )

    def update(self, dt, env, events):
        # cross-inject G1 → G2 and G2 → G1
        env["dopamine"]  = self.G1.Dopamine
        env["cortisol"]  = self.G1.Cortisol
        env["melatonin"] = self.G1.Melatonin
        env["insulin"]   = self.G2.Insulin
        env["estrogen"]  = self.G2.Estrogen
        env["leptin"]    = self.G2.Leptin

        self.G1.update(dt, env, events)
        self.G2.update(dt, env, events)
        self.G3.update(dt, env)
    def get_all_hormones(self):
        """
        Returns a flat dictionary of all hormone levels across G1, G2, G3.
        Format:
            {
                "Dopamine": 0.5,
                "Serotonin": 0.5,
                ...
            }
        """
        hormones = {}

        for group in [self.G1, self.G2, self.G3]:
            for name, value in vars(group).items():
                hormones[name] = value

        return hormones
    def get_all_hormones_grouped(self):

        return {
            "G1": vars(self.G1).copy(),
            "G2": vars(self.G2).copy(),
            "G3": vars(self.G3).copy()
        }
    def is_alive(self):
        return (
                self.G2.Insulin       > 0.01
                and (self.G2.Glucagon > 0.01 or self.G3.T3 > 0.01)
                and (self.G1.Dopamine > 0.01 or self.G1.Serotonin > 0.01)
        )

    def necessities(self, thresholds=None):
        if thresholds is None:
            thresholds = {
                "hunger":     ("Ghrelin",       "high", 0.60),
                "thirst":     ("Vasopressin",   "high", 0.55),
                "rest":       ("Melatonin",     "high", 0.50),
                "threat":     ("Adrenaline",    "high", 0.40),
                "social":     ("Oxytocin",      "low",  0.20),
                "energy":     ("Insulin",       "low",  0.25),
                "stress":     ("Cortisol",      "high", 0.55),
                "depression": ("Serotonin",     "low",  0.20),
                "pain":       ("Prostaglandins","high", 0.40),
            }
        active = []
        for necessity, (hormone, direction, threshold) in thresholds.items():
            value = (getattr(self.G1, hormone, None)
                     or getattr(self.G2, hormone, None)
                     or getattr(self.G3, hormone, None))
            if value is None:
                continue
            if direction == "high" and value >= threshold:
                active.append((necessity, round(value, 3)))
            elif direction == "low" and value <= threshold:
                active.append((necessity, round(value, 3)))
        return active

    def state(self):
        return {
            "G1":          self.G1.__dict__,
            "G2":          self.G2.__dict__,
            "G3":          self.G3.__dict__,
            "alive":       self.is_alive(),
            "necessities": self.necessities()
        }
    


# ============================================================
# SIMULATION
# ============================================================

if __name__ == "__main__":
    isv = InternalStateVector()

    env = {
        "activity_level": 0.0,   # sitting idle
        "darkness":       0.0,
        "light":          1.0,
        "pain_level":     0.0,
        "safety_level":   0.0,
        "food_intake":    0.0,   # not eating
        "blood_sugar":    0.5,
        "fat_reserves":   0.5,
        "hydration":      0.0,
        "blood_pressure": 0.5,
        "oxygen_level":   0.8,
        "damage_level":   0.0,
        "sleep_depth":    0.0,
    }

    events = {
        "success": 0, "failure": 0, "threat": 0,
        "social_success": 0, "social_bond": 0, "dominance": 0,
    }

    print("=== Simulating agent sitting idle ===\n")
    print(f"{'Tick':>6} | {'Dopamine':>8} | {'Cortisol':>8} | "
          f"{'Ghrelin':>7} | {'Serotonin':>9} | {'Vasopressin':>11} | Alive")
    print("-" * 75)

    for tick in range(36000000):
        isv.update(dt=0.1, env=env, events=events)
        if tick % 50 == 0:
            print(f"{tick:>6} | "
                  f"{isv.G1.Dopamine:>8.3f} | "
                  f"{isv.G1.Cortisol:>8.3f} | "
                  f"{isv.G2.Ghrelin:>7.3f} | "
                  f"{isv.G1.Serotonin:>9.3f} | "
                  f"{isv.G2.Vasopressin:>11.3f} | "
                  f"{isv.is_alive()}")
        if not isv.is_alive():
            print(f"\nAgent died at tick {tick}")
            break

    print("\n=== Active Necessities ===")
    for n, v in isv.necessities():
        print(f"  {n:>12}: {v:.3f}")
