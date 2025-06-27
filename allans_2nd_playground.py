from allans_playground import SimHubb2P

import matplotlib.pyplot as plt

def plot_states(L, U, states_to_plot):
    system = SimHubb2P(L=L, U=U)
    
    # Plot requested states
    figs = []
    for k in range(states_to_plot):
        fig = system.plot_eigenstate(k)
        fig.suptitle(f'L={L}, U={U}, State {k}', y=1.05)
        figs.append(fig)
    
    plt.show()
    return figs

if __name__ == "__main__":
    # Example usage:
    plot_states(L=15, U=4.0, states_to_plot=7)
