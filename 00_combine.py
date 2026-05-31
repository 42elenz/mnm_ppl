import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Directories
SOURCE_DIR = "figures/advanced"
OUTPUT_DIR = "figures/00_main_final_tables_figures/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The four ROC curves we generated in our loop
images = [
    "comprehensive_incremental_value_roc_all.png",  # Panel a (All Patients)
    "comprehensive_incremental_value_roc_ssd.png",  # Panel b (Schizophrenia Spectrum)
    "comprehensive_incremental_value_roc_bd.png",   # Panel c (Bipolar Disorder)
    "comprehensive_incremental_value_roc_mdd.png"   # Panel d (Major Depression)
]

# Set up the 2x2 grid (16x16 is perfect to keep text legible and crisp)
fig, axes = plt.subplots(
    2, 2, 
    figsize=(16, 16), 
    gridspec_kw={'height_ratios': [1, 1]} # Uniform height
)

panel_labels = ["a", "b", "c", "d"]
panel_titles = [
    "All Patients vs. HC", 
    "SSD vs. HC", 
    "BD vs. HC", 
    "MDD vs. HC"
]

for i, img_name in enumerate(images):
    ax = axes[i // 2, i % 2]
    img_path = os.path.join(SOURCE_DIR, img_name)
    
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)  # Keeps aspect ratio fully intact
    else:
        print(f"Warning: {img_path} not found.")
        
    ax.axis('off')  # Hide grid axes for clean image rendering
    
    # Place panel labels (a, b, c, d) in the top-left corner
    ax.text(-0.02, 1.02, panel_labels[i], transform=ax.transAxes, 
            fontsize=24, va='bottom', ha='right')

# Use tight_layout to eliminate excess white spaces between panels
plt.tight_layout(pad=1.0)

save_path = os.path.join(OUTPUT_DIR, "combined_comprehensive_roc_grid.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nSaved combined ROC results grid to: {save_path}")
plt.show()