"""Fisica avanzata del multirotore su fastuav.physics (Python puro, senza Isaac Sim).

Si attiva con `MultirotorConfig.physics` (vedi `config.PhysicsConfig`). Strati:
fastuav.physics (formule) -> questo pacchetto (drone montato, gemello offline)
-> `vehicles.multirotor.Multirotor` (forze applicate in Isaac).

Questo __init__ resta vuoto: `config` deve potersi importare senza fastuav.
"""
