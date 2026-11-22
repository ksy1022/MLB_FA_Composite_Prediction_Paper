import matplotlib.pyplot as plt
import numpy as np

roles = ['Hitter', 'Starting Pitcher', 'Relief Pitcher']
accuracy = [0.783, 0.723, 0.898]
f1 = [0.618, 0.308, 0.000]

x = np.arange(len(roles))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
bars2 = ax.bar(x + width/2, f1, width, label='High-class F1')

ax.set_ylabel('Score')
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(roles)
ax.set_title('Test Accuracy and High-Class F1 of Selected Binary Composite Models')
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom'
        )

plt.tight_layout()
plt.savefig('Figure3_Binary_Composite_Models.png', dpi=300, bbox_inches='tight')
plt.show()
