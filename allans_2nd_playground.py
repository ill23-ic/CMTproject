from allans_playground import SimHubb2P

import matplotlib.pyplot as plt

def plot_eigenstate_image(state_idx):
    system = SimHubb2P(L=5)
    fig = system.plot_eigenstate(state_idx)

    fig.suptitle(f'Hubbard Model Eigenstate {state_idx}', y=1.05)

    return fig


if __name__ == "__main__":
    for k in range (5):
        plot_eigenstate_image(k)
    plt.show()
