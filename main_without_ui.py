from time import sleep
from simulation import Simulation
from threading import Thread


def do_stuff() -> None:
    sleep(1)

    simulation.set_mu(0.5)

    sleep(10)

    simulation.set_frequency(0)

    simulation.throw_particles(1000)
    sleep(1)

    simulation.throw_particles(1000)
    sleep(1)

    simulation.throw_particles(1000)
    sleep(1)


Thread(target=do_stuff).start()
simulation = Simulation()
simulation.start()
