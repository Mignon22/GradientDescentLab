import matplotlib.pyplot as plt
import numpy as np

def plot_loss_and_lr(loss_history_train, loss_history_test, lr_history = None, min_index = None, min_test_loss = None, title_main = 'Experiment : Loss vs Learning Rate', subtitle = '', figsize = (8, 5.5), save_path = None):
    loss_history_train = np.array(loss_history_train)
    loss_history_test = np.array(loss_history_test)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize = figsize)
    ax1.set_title(title_main, fontsize = 14, fontweight = 'bold', pad = 24)
    if subtitle:
        fig.text(0.5, 0.893, subtitle, ha = 'center', fontsize = 12, style = 'italic')
    
    # 主軸
    color1 = '#1f77bc'
    color2 = '#ff7f0e'
    color3 = '#003366'
    ax1.set_xlabel('Iterations', fontsize = 14, fontweight = 'bold')
    ax1.set_ylabel('Loss', color = color1, fontsize = 14, fontweight = 'bold')
    ax1.plot(loss_history_train, label = 'Train Loss', color = color1, linewidth = 3, alpha = 0.85, zorder = 3)
    ax1.plot(loss_history_test, label = 'Test Loss', color = color2, linewidth = 3, linestyle = 'dashed', alpha = 0.85, zorder = 3)
    ax1.tick_params(axis = 'y', labelcolor = color1)

    ax1.plot(min_index, min_test_loss, 'o', markersize = 5, label = 'Min Test Loss', color = color3, zorder = 4)

    bbox_props = dict(boxstyle = 'round,pad = 0.4', fc = 'white', lw = 0.8, alpha = 0.85)
    ax1.annotate(
        f'Min : {min_test_loss:.4f} @ Iter {min_index}',
        xy = (min_index, min_test_loss),
        xytext = (max(min_index - len(loss_history_test) * 0.2, 0), min_test_loss + (loss_history_test.max() - loss_history_test.min()) * 0.2),
        textcoords = 'data',
        arrowprops = dict(arrowstyle = '-|>', color = color3, lw = 1.2, zorder = 4),
        fontsize = 12,
        color = color3,
        bbox = bbox_props,
        zorder = 4
    )
    ax1.legend(loc = 'upper right', fontsize = 11, framealpha = 0.9)
    ax1.grid(True, linestyle = 'dashed', linewidth = 0.5, alpha = 0.4, zorder = 0)

    # 副軸
    if lr_history is not None:
        lr_history = np.array(lr_history)
        ax2 = ax1.twinx()
        color4 = '#2ca02c'
        ax2.set_ylabel('Learning Rate', color = color4, fontsize = 14, fontweight = 'bold')
        ax2.plot(lr_history, label = 'Learning Rate', color = color4, linestyle = 'dotted', linewidth = 2, zorder = 1)
        ax2.tick_params(axis = 'y', labelcolor = color4)
    
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi = 720, bbox_inches = 'tight')
    plt.show()
