# -*- coding: utf-8 -*-
"""
This script analyzes embedding characteristics and generates optimization recommendations.

"""

import sys
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List
from core.config.settings import Settings
from core.services.embedding_service import EmbeddingService

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize system components
settings = Settings()
embedding_service = EmbeddingService(settings)


async def analyze_embeddings(sample_texts: List[str]):
    print(f"Generating embeddings for {len(sample_texts)} texts...")

    embedding_results = await embedding_service.create_embeddings_batch(sample_texts)
    embeddings = np.array([result.embedding for result in embedding_results])

    # Calculate statistics
    stats = {
        "dimension": embeddings.shape[1],
        "mean_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
        "std_norm": np.std(np.linalg.norm(embeddings, axis=1)),
        "mean_value": np.mean(embeddings),
        "std_value": np.std(embeddings),
        "min_value": np.min(embeddings),
        "max_value": np.max(embeddings),
    }

    return embeddings, stats, embedding_results


def generate_recommendations(embedding_stats):
    recommendations = []

    # Embedding recommendations
    if embedding_stats["std_norm"] > 0.1:
        recommendations.append(
            "High variance in embedding norms - consider normalization"
        )

    return recommendations


# Sample texts for embedding analysis
sample_texts = [
    "Εκουσία δικαιοδοσία. Ανακοπή κατά κληρονομητηρίου 824ΚΠολΔ, η οποία κάνει δεκτή εν μέρει την ανακοπή ως προς τον δεύτερο θάνατο (πατέρα), ενώ για θάνατο του παππού την απέρριψε ως μη νόμιμη λόγω ότι ο θάνατος επήλθε το 1918 και εφαρμογή του βυζαντινορωμαϊκού δικαίου (άρθρ 92 ΕισΑκ).",
    "Τηλεφωνικές κλήσεις για σκοπούς απευθείας προώθησης προϊόντων ή υπηρεσιών και για διαφημιστικούς σκοπούς. Αποζημίωση ποσού 4.000,00 ευρώ, ως χρηματική ικανοποίηση λόγω ηθικής βλάβης νομιμότοκα ήτοι από το 2022. Παράνομη επεξεργασία προσωπικών δεδομένων. Προσωρινά εκτελεστή η απόφαση.",
    "Επιβολή ποινής φυλάκισης 3 ετών για την κλοπή και χρηματικής ποινής 90 ημερησίων μονάδων προς 3 ευρώ εκάστη για την χρήση πλαστού. Μετατροπή της ποινής φυλάκισης σε παροχή κοινωφελούς εργασίας με εφαρμογή της ευμενέστερης διάταξης που ίσχυσε από την τέλεση της πράξης μέχρι την αμετάκλητη εκδίκασή της.",
    "Το Γενικό Πολεοδομικό Σχέδιο, το οποίο συγκροτεί το πρώτο επίπεδο πολεοδομικού σχεδιασμού, έχει, καταρχήν, ως πεδίο αναφοράς την κτηματική περιφέρεια ορισμένου Οργανισμού Τοπικής Αυτοδιοίκησης, δηλαδή υποδιαίρεση του χώρου που καθορίζεται ως πεδίο άσκησης των αρμοδιοτήτων των πρωτοβάθμιων τοπικών αρχών, κατά τα ειδικότερα οριζόμενα στο Σύνταγμα και τον νόμο. Ο ορισμός, όμως, από τον νόμο της περιφέρειας ενός πρωτοβαθμίου ΟΤΑ ως περιοχής αναφοράς του ΓΠΣ δεν έχει την αντίστροφη έννοια του ακριβούς καθορισμού ή τροποποίησης των ορίων του ΟΤΑ μέσω της έγκρισης του ΓΠΣ, διότι ο καθορισμός των ορίων και η τροποποίηση αυτών επιχειρείται, σύμφωνα με τον νόμο, από άλλα διοικητικά και δικαστικά όργανα και με άλλη ειδικώς οριζόμενη διαδικασία",
    "Τροχαίο ατύχημα οφειλόμενο σε αποκλειστική ευθύνη του δεύτερου εναγόμενου της Α αγωγής, ο οποίος δεν οδηγούσε με σύνεση και με διαρκώς τεταμένη την προσοχή στην οδήγηση και πραγματοποίησε αιφνιαδιαστικά αλλαγή λωρίδας κυκλοφορίας, χωρίς προηγουμένως να βεβαιωθεί ότι μπορεί να πράξει τούτο άνευ κινδύνου ή παρακωλύσεως των λοιπών χρησιμοποιούντων την οδό. Το όχημα του δε δεν ήταν εφοδιασμένο με έναν τουλάχιστον καθρέπτη, τοποθετημένο σε θέση η οποία να εξασφαλίσει την ορατότητα της οδού πίσω από το όχημα. Η εξετασθείσα στο ακροατήριο μάρτυρας απόδειξης της Β αγωγής  στην κατάθεση της ήταν ασαφής. Το Δικαστήριο απέρριψε ως ουσιαστικά αβάσιμους τους ισχυρισμούς του ενάγοντος της Β' αγωγής, καθόσον οι σχετικές απαγωγικές του αιτιάσεις στηρίζονται αποκλειστικά στην ένορκη κατάθεση της μάρτυρος απόδειξης της εν λόγω αγωγής, η οποία, όμως, κρίθηκε μη πειστική ως προς το συγκεκριμένο ζήτημα. Πρόκληση υλικών ζημιών σε μοτοσικλέτα.  Επιδίκαση στον ενάγοντα, για την αποκατάσταση των υλικών ζημιών που υπέστη η μοτοσικλέτα του, ποσού το οποίο περιλαμβάνει τόσο τη δαπάνη στην οποία ήδη υποβλήθηκε όσο και την εκτιμώμενη δαπάνη για τις εναπομένουσες εργασίες και ανταλλακτικά. Ηθική βλάβη λόγω του τραυματισμού.",
]

embeddings, embedding_stats, embedding_results = asyncio.run(
    analyze_embeddings(sample_texts)
)
print(f"\n✅ Generated embeddings with dimension {embedding_stats['dimension']}")

# Visualize embedding characteristics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Embedding norms
norms = np.linalg.norm(embeddings, axis=1)
axes[0, 0].bar(range(len(norms)), norms)
axes[0, 0].set_title("Embedding Vector Norms")
axes[0, 0].set_xlabel("Text Index")
axes[0, 0].set_ylabel("L2 Norm")

# Embedding value distribution
axes[0, 1].hist(embeddings.flatten(), bins=50, alpha=0.7)
axes[0, 1].set_title("Embedding Value Distribution")
axes[0, 1].set_xlabel("Embedding Value")
axes[0, 1].set_ylabel("Frequency")

# Similarity matrix
similarity_matrix = np.dot(embeddings, embeddings.T)
im = axes[1, 0].imshow(similarity_matrix, cmap="viridis")
axes[1, 0].set_title("Cosine Similarity Matrix")
axes[1, 0].set_xlabel("Text Index")
axes[1, 0].set_ylabel("Text Index")
plt.colorbar(im, ax=axes[1, 0])

# Processing time vs token count
token_counts = [result.token_count for result in embedding_results]
processing_times = [result.processing_time for result in embedding_results]
axes[1, 1].scatter(token_counts, processing_times)
axes[1, 1].set_title("Processing Time vs Token Count")
axes[1, 1].set_xlabel("Token Count")
axes[1, 1].set_ylabel("Processing Time (seconds)")

plt.tight_layout()
plt.show()

# Display embedding statistics
print("\nEmbedding Statistics:")
for key, value in embedding_stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")

# Generate and display recommendations
recommendations = generate_recommendations(embedding_stats)

# Export embedding analysis results
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "embedding_statistics.json", "w") as f:
    json.dump(embedding_stats, f, indent=2)
print(f"Saved embedding statistics to {output_dir / 'embedding_statistics.json'}")
