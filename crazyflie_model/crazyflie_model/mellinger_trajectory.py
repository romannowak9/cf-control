import numpy as np


class MellingerTrajectory:
    def __init__(self, p0, pT, yaw0, yawT, T_flight):
        """
        Faza Generowania: Oblicza parametry i współczynniki trajektorii raz.
        Zgodne z zasadą Spatial i Temporal Scaling z artykułu Mellingera (Sekcja V-A).
        """
        self.p0 = np.array(p0, dtype=float)
        self.pT = np.array(pT, dtype=float)
        self.yaw0 = float(yaw0)
        self.yawT = float(yawT)
        self.T_flight = float(T_flight)

        # Różnice przestrzenne (Spatial Scaling)
        self.delta_p = self.pT - self.p0
        self.delta_yaw = self.yawT - self.yaw0

        # 1. POZYCJA: Wielomian 7. stopnia (Minimum Snap, k_r = 4)
        # Spełnia warunki: p(0)=0, p(1)=1 oraz pochodne 1, 2, 3 równe 0 na obu końcach.
        # Przechowujemy niezerowe współczynniki dla potęg: [tau^4, tau^5, tau^6, tau^7]
        self.c_pos = np.array([35.0, -84.0, 70.0, -20.0])

        # 2. YAW: Wielomian 5. stopnia (Minimum Acceleration, k_psi = 2)
        # Zgodnie z artykułem, dla yaw k_psi = 2. Przy założeniu zerowej prędkości i
        # zerowego przyspieszenia kątowego na końcach odcinka, analitycznym rozwiązaniem
        # optymalizacyjnym jest wielomian 5. stopnia (tzw. minimum jerk/acceleration profil).
        # Przechowujemy niezerowe współczynniki dla potęg: [tau^3, tau^4, tau^5]
        self.c_yaw = np.array([10.0, -15.0, 6.0])

    def evaluate(self, t_eval):
        """
        Faza Wykonania: Szybka ewaluacja stanu trajektorii w pętli kontrolera.
        Przyjmuje: t_eval (czas od rozpoczęcia manewru w sekundach).
        Zwraca: pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc
        """
        # Jeśli czas przekroczył zaplanowany czas lotu, dron wykonuje zawis w punkcie docelowym
        if t_eval >= self.T_flight:
            return (
                self.pT,
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                self.yawT,
                0.0,
                0.0,
            )

        t = max(0.0, t_eval)
        tau = t / self.T_flight  # Czas bezwymiarowy [0, 1]

        # --- EWALUACJA POZYCJI (Wielomian 7. stopnia - Minimum Snap) ---
        p_tau = (
            self.c_pos[0] * tau**4
            + self.c_pos[1] * tau**5
            + self.c_pos[2] * tau**6
            + self.c_pos[3] * tau**7
        )

        dp_tau = (
            4 * self.c_pos[0] * tau**3
            + 5 * self.c_pos[1] * tau**4
            + 6 * self.c_pos[2] * tau**5
            + 7 * self.c_pos[3] * tau**6
        )

        ddp_tau = (
            12 * self.c_pos[0] * tau**2
            + 20 * self.c_pos[1] * tau**3
            + 30 * self.c_pos[2] * tau**4
            + 42 * self.c_pos[3] * tau**5
        )

        dddp_tau = (
            24 * self.c_pos[0] * tau
            + 60 * self.c_pos[1] * tau**2
            + 120 * self.c_pos[2] * tau**3
            + 210 * self.c_pos[3] * tau**4
        )

        ddddp_tau = (
            24 * self.c_pos[0]
            + 120 * self.c_pos[1] * tau
            + 360 * self.c_pos[2] * tau**2
            + 840 * self.c_pos[3] * tau**3
        )

        # Skalowanie czasowe i przestrzenne (Temporal & Spatial Scaling) dla pozycji
        pos = self.p0 + self.delta_p * p_tau
        vel = self.delta_p * dp_tau / self.T_flight
        acc = self.delta_p * ddp_tau / (self.T_flight**2)
        jerk = self.delta_p * dddp_tau / (self.T_flight**3)
        snap = self.delta_p * ddddp_tau / (self.T_flight**4)

        # --- EWALUACJA YAW (Wielomian 5. stopnia - Minimum Acceleration, k_psi = 2) ---
        p_yaw_tau = self.c_yaw[0] * tau**3 + self.c_yaw[1] * tau**4 + self.c_yaw[2] * tau**5

        dp_yaw_tau = (
            3 * self.c_yaw[0] * tau**2 + 4 * self.c_yaw[1] * tau**3 + 5 * self.c_yaw[2] * tau**4
        )

        ddp_yaw_tau = (
            6 * self.c_yaw[0] * tau + 12 * self.c_yaw[1] * tau**2 + 20 * self.c_yaw[2] * tau**3
        )

        # Skalowanie czasowe i przestrzenne dla kąta obrotu yaw
        yaw = self.yaw0 + self.delta_yaw * p_yaw_tau
        yaw_dot = self.delta_yaw * dp_yaw_tau / self.T_flight
        yaw_acc = self.delta_yaw * ddp_yaw_tau / (self.T_flight**2)

        return pos, vel, acc, jerk, snap, yaw, yaw_dot, yaw_acc
