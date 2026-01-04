import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.patches as patches

def generate_academic_comparison():
    # 1. Definicja punktów (układ "Latawiec" wymuszający różnicę w przekątnej)
    points = np.array([
        [4, 8],  # 0: Wierzchołek górny
        [2, 4],  # 1: Wierzchołek lewy
        [6, 4],  # 2: Wierzchołek prawy
        [4, 2],  # 3: Wierzchołek dolny (wewnątrz)
        [1, 1],  # 4: Tło
        [7, 1],  # 5: Tło
        [4, 9.5] # 6: Tło
    ])

    # 2. Ręczna definicja połączeń (aby mieć pewność co do kształtu siatki)
    
    # Konfiguracja Delaunaya (Optymalna - łączy punkty 1 i 2)
    # Maksymalizuje minimalne kąty
    delaunay_simplices = [
        [0, 1, 2], # Górny
        [1, 2, 3], # Dolny
        [1, 3, 4], [2, 3, 5], [0, 1, 6], [0, 2, 6] # Tło
    ]

    # Konfiguracja Dowolna (Suboptymalna - łączy punkty 0 i 3)
    # Tworzy wąskie trójkąty
    general_simplices = [
        [0, 1, 3], # Lewy wąski
        [0, 2, 3], # Prawy wąski
        [1, 3, 4], [2, 3, 5], [0, 1, 6], [0, 2, 6] # Tło
    ]

    # --- RYSOWANIE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def plot_mesh(ax, simplicies, mode):
        # Rysowanie siatki bazowej (szare linie)
        ax.triplot(points[:,0], points[:,1], simplicies, color='gray', linestyle='-', alpha=0.4, linewidth=1)
        # Punkty
        ax.plot(points[:,0], points[:,1], 'ko', markersize=4, zorder=5) 
        
        # Etykiety punktów - opcjonalnie, można usunąć w wersji finalnej
        # for i, p in enumerate(points):
        #    ax.text(p[0]+0.15, p[1], f"$P_{{{i}}}$", fontsize=9)

        if mode == "general":
            # Wyróżnienie przekątnej dla triangulacji ogólnej (0 -> 3)
            ax.plot([points[0,0], points[3,0]], [points[0,1], points[3,1]], 
                    color='#D62728', linestyle='--', linewidth=2)
            
            # Delikatne podświetlenie wąskich trójkątów
            t1 = plt.Polygon([points[0], points[1], points[3]], color='#D62728', alpha=0.08)
            t2 = plt.Polygon([points[0], points[2], points[3]], color='#D62728', alpha=0.08)
            ax.add_patch(t1)
            ax.add_patch(t2)
            
            

        elif mode == "delaunay":
            # Wyróżnienie przekątnej dla triangulacji Delaunaya (1 -> 2)
            ax.plot([points[1,0], points[2,0]], [points[1,1], points[2,1]], 
                    color='#2CA02C', linestyle='-', linewidth=2)
            
            # Podświetlenie trójkątów Delaunaya
            t1 = plt.Polygon([points[0], points[1], points[2]], color='#2CA02C', alpha=0.08)
            t2 = plt.Polygon([points[1], points[2], points[3]], color='#2CA02C', alpha=0.08)
            ax.add_patch(t1)
            ax.add_patch(t2)


        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=9, frameon=True)
        ax.axis('off') # Wyłączenie osi dla czystości rysunku

    # Generowanie wykresów
    plot_mesh(ax1, general_simplices, mode="general")
    plot_mesh(ax2, delaunay_simplices, mode="delaunay")

    plt.tight_layout()
    plt.savefig("porownanie_akademickie.png", dpi=300, bbox_inches='tight')
    print("Wygenerowano plik: porownanie_akademickie.png")
    plt.show()

generate_academic_comparison()