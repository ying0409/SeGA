import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def user_embedding_visualization(embeddings, labels, name, save_path='figure/'):
    print("Embedding visualization")
    draw_id = torch.load("/home/yychang/Sega/datasets/processed_data/sample/sample_draw_id_1.pt", map_location='cpu')
    print("# of nodes:", len(draw_id))

    embeddings = embeddings[draw_id]
    labels = labels[draw_id]

    tsne = TSNE(n_components=2, random_state=42, perplexity=15) # perplexity=7
    embedded_embeddings = tsne.fit_transform(embeddings)

    classes_embeddings = [embedded_embeddings[labels == class_label] for class_label in set(labels)]

    colors = ['mediumseagreen', 'red', 'blue']
    labels_names = ['Normal User', 'Bot', 'Troll']
    for i, class_embeddings in enumerate(classes_embeddings):
        plt.scatter(class_embeddings[:, 0], class_embeddings[:, 1], c=colors[i], label=labels_names[i])
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')

    plt.axis('off')
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path + name + ".png")
    plt.close()