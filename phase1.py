from sympy import symbols, simplify, expand, Rational

eps = symbols('epsilon', positive=True)

# Each entry: (L, U, probability)
states = [(0, 150, Rational(1))]

for t in range(7):
    E_x = 0
    P_aplus = 0
    for L, U, p in states:
        x = (L + U) // 2
        E_x = E_x + p * x
        if x <= 100:
            p_plus = (1 + eps) / 2
        else:
            p_plus = (1 - eps) / 2
        P_aplus = P_aplus + p * p_plus

    print(f"t={t}: E[x_t] = {expand(E_x)}")
    print(f"      P(a_t=+1) = {expand(P_aplus)}")
    print(f"      num states = {len(states)}")
    print()

    # Branch
    new_states = []
    for L, U, p in states:
        x = (L + U) // 2
        if x <= 100:
            p_plus = (1 + eps) / 2
        else:
            p_plus = (1 - eps) / 2
        p_minus = 1 - p_plus
        new_states.append((x, U, p * p_plus))
        new_states.append((L, x, p * p_minus))
    states = new_states
