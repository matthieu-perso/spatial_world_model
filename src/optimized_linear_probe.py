
import h5py
import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pickle
import time

class SpatialLinearProbe:
    def __init__(self, hdf5_path, output_dir='results', batch_size=1024, test_size=0.2, gpu=True):
        """
        Initialize the SpatialLinearProbe class.

        Args:
            hdf5_path (str): Path to the HDF5 file containing embeddings
            output_dir (str): Directory to save outputs
            batch_size (int): Batch size for GPU processing
            test_size (float): Proportion of data to use for testing
            gpu (bool): Whether to use GPU acceleration
        """
        self.hdf5_path = hdf5_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print(f"Using device: {self.device}")

    def load_data(self):
        """Load data from HDF5 file"""
        start_time = time.time()
        print("Loading data from HDF5 file...")

        with h5py.File(self.hdf5_path, "r") as f:
            # Load the sentences
            self.sentences = [s.decode('utf-8') for s in f["sentences"][:]]

            # Load embeddings for each layer
            self.layer8 = torch.tensor(f["layer_8"][:], dtype=torch.float32)
            self.layer16 = torch.tensor(f["layer_16"][:], dtype=torch.float32)
            self.layer24 = torch.tensor(f["layer_24"][:], dtype=torch.float32)

        print(f"Loaded {len(self.sentences)} sentences and their embeddings")
        print(f"Layer 8 embeddings shape: {self.layer8.shape}")
        print(f"Layer 16 embeddings shape: {self.layer16.shape}")
        print(f"Layer 24 embeddings shape: {self.layer24.shape}")
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    def parse_sentences(self):
        """Parse sentences into object1, relation, object2 triplets"""
        start_time = time.time()
        print("Parsing sentences into triplets...")

        # Primary pattern for "The X is Y the Z." format
        pattern = r"The (.*?) is (.*?) the (.*?)\."

        triplets = []
        valid_indices = []

        for i, sentence in enumerate(self.sentences):
            match = re.search(pattern, sentence)
            if match:
                obj1, relation, obj2 = match.groups()
                triplets.append((obj1.strip(), relation.strip(), obj2.strip()))
                valid_indices.append(i)
            else:
                # Secondary pattern for relations without "the" (like "on", "facing")
                # Look for patterns like "The table is on the chair." or "The table is facing the lamp."
                alt_pattern = r"The (.*?) is (.*) (chair|lamp|table)\."
                match = re.search(alt_pattern, sentence)
                if match:
                    obj1, relation, obj2 = match.groups()
                    triplets.append((obj1.strip(), relation.strip(), obj2.strip()))
                    valid_indices.append(i)
                else:
                    # Last attempt with very general pattern
                    try:
                        # Split the sentence
                        # "The table is connected to the lamp." → ["The", "table", "is", "connected", "to", "the", "lamp."]
                        words = sentence.split()
                        if len(words) >= 5 and words[0].lower() == "the" and words[2].lower() == "is":
                            obj1 = words[1]

                            # Find the last occurrence of "the" to locate obj2
                            if "the" in words[3:]:
                                last_the_idx = len(words) - 1 - words[::-1].index("the")
                                obj2 = words[last_the_idx+1].rstrip('.')
                                relation = " ".join(words[3:last_the_idx])
                                triplets.append((obj1.strip(), relation.strip(), obj2.strip()))
                                valid_indices.append(i)
                            else:
                                # If no "the" found, assume the last word is the object
                                obj2 = words[-1].rstrip('.')
                                relation = " ".join(words[3:-1])
                                triplets.append((obj1.strip(), relation.strip(), obj2.strip()))
                                valid_indices.append(i)
                        else:
                            print(f"Failed to parse: {sentence}")
                    except Exception as e:
                        print(f"Error parsing: {sentence}, Error: {e}")

        self.triplets = triplets
        self.valid_indices = valid_indices

        print(f"Successfully parsed {len(triplets)} triplets")
        print(f"Parsing completed in {time.time() - start_time:.2f} seconds")

        # Display a few parsed triplets for verification
        print("\nSample triplets:")
        for i in range(min(5, len(triplets))):
            print(f"{i}: {triplets[i]}")

    def encode_triplets(self):
        """Encode triplets for model training"""
        start_time = time.time()
        print("Encoding triplets...")

        # Extract separate components
        objects1, relations, objects2 = zip(*self.triplets)

        # Filter out empty strings
        objects1 = [obj if obj else "UNKNOWN" for obj in objects1]
        relations = [rel if rel else "UNKNOWN" for rel in relations]
        objects2 = [obj if obj else "UNKNOWN" for obj in objects2]

        # Encode each component
        self.obj1_encoder = LabelEncoder()
        self.rel_encoder = LabelEncoder()
        self.obj2_encoder = LabelEncoder()

        self.obj1_labels = self.obj1_encoder.fit_transform(objects1)
        self.rel_labels = self.rel_encoder.fit_transform(relations)
        self.obj2_labels = self.obj2_encoder.fit_transform(objects2)

        # Create mapping dictionaries for later analysis
        self.obj1_mapping = dict(zip(self.obj1_encoder.classes_, range(len(self.obj1_encoder.classes_))))
        self.rel_mapping = dict(zip(self.rel_encoder.classes_, range(len(self.rel_encoder.classes_))))
        self.obj2_mapping = dict(zip(self.obj2_encoder.classes_, range(len(self.obj2_encoder.classes_))))

        print(f"Found {len(self.obj1_mapping)} unique objects as subject")
        print(f"Found {len(self.rel_mapping)} unique relations")
        print(f"Found {len(self.obj2_mapping)} unique objects as object")

        # Print some of the unique relations
        print("\nSample relations:")
        sample_relations = list(self.rel_mapping.keys())[:10]
        for i, rel in enumerate(sample_relations):
            print(f"{i}: {rel}")

        print(f"Encoding completed in {time.time() - start_time:.2f} seconds")

    def prepare_data_split(self, layer_data):
        """Prepare a single train/test split for all labels"""
        # Filter embeddings to keep only valid indices
        X = layer_data[self.valid_indices]

        # Create a single train/test split for all labels
        X_train, X_test, y_obj1_train, y_obj1_test, y_rel_train, y_rel_test, y_obj2_train, y_obj2_test = train_test_split(
            X,
            self.obj1_labels,
            self.rel_labels,
            self.obj2_labels,
            test_size=self.test_size,
            random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        y_obj1_train_tensor = torch.tensor(y_obj1_train, dtype=torch.long)
        y_obj1_test_tensor = torch.tensor(y_obj1_test, dtype=torch.long)

        y_rel_train_tensor = torch.tensor(y_rel_train, dtype=torch.long)
        y_rel_test_tensor = torch.tensor(y_rel_test, dtype=torch.long)

        y_obj2_train_tensor = torch.tensor(y_obj2_train, dtype=torch.long)
        y_obj2_test_tensor = torch.tensor(y_obj2_test, dtype=torch.long)

        return {
            'X_train': X_train_tensor,
            'X_test': X_test_tensor,
            'y_obj1_train': y_obj1_train_tensor,
            'y_obj1_test': y_obj1_test_tensor,
            'y_rel_train': y_rel_train_tensor,
            'y_rel_test': y_rel_test_tensor,
            'y_obj2_train': y_obj2_train_tensor,
            'y_obj2_test': y_obj2_test_tensor
        }

    def create_loaders(self, split_data):
        """Create DataLoaders for training and evaluation"""
        # Create training datasets
        train_obj1_dataset = TensorDataset(split_data['X_train'], split_data['y_obj1_train'])
        train_rel_dataset = TensorDataset(split_data['X_train'], split_data['y_rel_train'])
        train_obj2_dataset = TensorDataset(split_data['X_train'], split_data['y_obj2_train'])

        # Create test datasets
        test_obj1_dataset = TensorDataset(split_data['X_test'], split_data['y_obj1_test'])
        test_rel_dataset = TensorDataset(split_data['X_test'], split_data['y_rel_test'])
        test_obj2_dataset = TensorDataset(split_data['X_test'], split_data['y_obj2_test'])

        # Create data loaders
        train_obj1_loader = DataLoader(train_obj1_dataset, batch_size=self.batch_size, shuffle=True)
        train_rel_loader = DataLoader(train_rel_dataset, batch_size=self.batch_size, shuffle=True)
        train_obj2_loader = DataLoader(train_obj2_dataset, batch_size=self.batch_size, shuffle=True)

        test_obj1_loader = DataLoader(test_obj1_dataset, batch_size=self.batch_size)
        test_rel_loader = DataLoader(test_rel_dataset, batch_size=self.batch_size)
        test_obj2_loader = DataLoader(test_obj2_dataset, batch_size=self.batch_size)

        return {
            'train': {
                'obj1': train_obj1_loader,
                'relation': train_rel_loader,
                'obj2': train_obj2_loader
            },
            'test': {
                'obj1': test_obj1_loader,
                'relation': test_rel_loader,
                'obj2': test_obj2_loader
            }
        }

    class LinearProbe(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    def train_probe(self, loader, input_dim, output_dim, epochs=5, lr=0.001):
        """Train a linear probe model"""
        model = self.LinearProbe(input_dim, output_dim).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print epoch results
            avg_loss = total_loss / len(loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        return model

    def evaluate_probe(self, model, loader):
        """Evaluate a trained probe model"""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, output_dict=True)

        return accuracy, report

    def extract_representations(self, model, num_classes, component_name):
        """Extract representations from model weights"""
        # Get the weights from the linear layer
        weights = model.linear.weight.data.cpu().numpy()

        if component_name == 'obj1':
            mapping = self.obj1_mapping
        elif component_name == 'relation':
            mapping = self.rel_mapping
        else:  # component == 'obj2'
            mapping = self.obj2_mapping

        # Create a mapping from class names to their weight vectors
        representations = {}
        for class_name, class_idx in mapping.items():
            if class_idx < num_classes:
                representations[class_name] = weights[class_idx]

        return representations

    def analyze_spatial_relations(self, representations):
        """Analyze the relationship between opposite spatial relations"""
        # Define pairs of opposite spatial relations
        opposite_pairs = [
            ("above", "below"),
            ("over", "under"),
            ("on top of", "beneath"),
            ("higher than", "lower than"),
            ("elevated above", "beneath"),
            ("to the left of", "to the right of"),
            ("in front of", "behind"),
            ("ahead of", "behind"),
            ("before", "behind"),
            ("inside", "outside"),
            ("within", "outside"),
            ("enclosed in", "outside"),
            ("close to", "far from"),
            ("near", "distant from"),
            ("next to", "far from"),
            ("adjacent to", "distant from"),
            ("beside", "away from"),
        ]

        relation_representations = representations['relation']

        results = []
        for rel1, rel2 in opposite_pairs:
            if rel1 in relation_representations and rel2 in relation_representations:
                vec1 = relation_representations[rel1]
                vec2 = relation_representations[rel2]

                # Calculate cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                # Calculate Euclidean distance
                distance = np.linalg.norm(vec1 - vec2)

                # Calculate angle in degrees
                angle = np.arccos(np.clip(similarity, -1.0, 1.0)) * 180 / np.pi

                results.append({
                    'relation1': rel1,
                    'relation2': rel2,
                    'cosine_similarity': similarity,
                    'euclidean_distance': distance,
                    'angle_degrees': angle
                })

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)

        if len(df) > 0:
            print("\nAnalysis of opposite spatial relations:")
            print(f"Average cosine similarity: {df['cosine_similarity'].mean():.4f}")
            print(f"Average Euclidean distance: {df['euclidean_distance'].mean():.4f}")
            print(f"Average angle (degrees): {df['angle_degrees'].mean():.2f}°")

            # Sort by angle (higher angle = more opposite)
            df_sorted = df.sort_values('angle_degrees', ascending=False)
            print("\nMost opposite relations (sorted by angle):")
            print(df_sorted[['relation1', 'relation2', 'angle_degrees']].head(5))

        return df

    def visualize_relation_geometry(self, representations, method='pca'):
        """Visualize the geometry of relation representations"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import numpy as np

        # Convert to numpy array to ensure shape attribute is available
        relation_vectors = np.array(list(representations['relation'].values()))
        relation_names = list(representations['relation'].keys())

        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            title = "PCA Visualization of Spatial Relations"
        else:  # tsne
            reducer = TSNE(n_components=2, perplexity=min(30, len(relation_vectors)-1), random_state=42)
            title = "t-SNE Visualization of Spatial Relations"

        # Transform data
        relation_vectors_2d = reducer.fit_transform(relation_vectors)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': relation_vectors_2d[:, 0],
            'y': relation_vectors_2d[:, 1],
            'relation': relation_names
        })

        # Create a categorical column for coloring
        df['category'] = 'Other'
        df.loc[df['relation'].str.contains('left|right'), 'category'] = 'Horizontal'
        df.loc[df['relation'].str.contains('above|below|over|under|top|beneath|higher|lower|elevated'), 'category'] = 'Vertical'
        df.loc[df['relation'].str.contains('front|behind|ahead|before|back'), 'category'] = 'Depth'
        df.loc[df['relation'].str.contains('inside|outside|within|enclosed'), 'category'] = 'Container'
        df.loc[df['relation'].str.contains('close|far|near|distant|next|adjacent|beside|away'), 'category'] = 'Proximity'

        # Create plot
        plt.figure(figsize=(12, 10))

        # Plot points with categorical colors
        categories = df['category'].unique()
        for category in categories:
            subset = df[df['category'] == category]
            plt.scatter(subset['x'], subset['y'], label=category, s=100, alpha=0.7)

        # Add relation names as labels
        for i, row in df.iterrows():
            plt.annotate(row['relation'], (row['x'], row['y']), fontsize=9)

        # Add opposite relation connections
        opposite_pairs = [
            ("above", "below"),
            ("over", "under"),
            ("on top of", "beneath"),
            ("higher than", "lower than"),
            ("to the left of", "to the right of"),
            ("in front of", "behind"),
            ("inside", "outside"),
            ("close to", "far from"),
        ]

        for rel1, rel2 in opposite_pairs:
            if rel1 in relation_names and rel2 in relation_names:
                idx1 = relation_names.index(rel1)
                idx2 = relation_names.index(rel2)
                plt.plot([relation_vectors_2d[idx1, 0], relation_vectors_2d[idx2, 0]],
                         [relation_vectors_2d[idx1, 1], relation_vectors_2d[idx2, 1]],
                         'r--', alpha=0.4)

        plt.title(title, fontsize=15)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        return plt

    def process_layer(self, layer_name, layer_data, epochs=5):
        """Process a single layer"""
        print(f"\n{'='*50}")
        print(f"Processing {layer_name}")
        print(f"{'='*50}")

        # Prepare data split
        split_data = self.prepare_data_split(layer_data)

        # Create data loaders
        loaders = self.create_loaders(split_data)

        # Track results and models
        results = {}
        models = {}
        representations = {}

        # Train and evaluate probes for each component
        for component, num_classes in [
            ('obj1', len(self.obj1_mapping)),
            ('relation', len(self.rel_mapping)),
            ('obj2', len(self.obj2_mapping))
        ]:
            print(f"\nTraining probe for {component} classification ({num_classes} classes)...")

            # Get input dimension
            input_dim = layer_data.shape[1]

            # Train probe
            model = self.train_probe(
                loaders['train'][component],
                input_dim=input_dim,
                output_dim=num_classes,
                epochs=epochs
            )

            # Evaluate probe
            accuracy, report = self.evaluate_probe(model, loaders['test'][component])
            print(f"{component} probe accuracy: {accuracy:.4f}")

            # Store results
            results[component] = {
                'accuracy': accuracy,
                'report': report
            }

            # Store model
            models[component] = model

            # Extract representations
            component_representations = self.extract_representations(model, num_classes, component)
            representations[component] = component_representations

        # Save results
        results_path = os.path.join(self.output_dir, f"{layer_name}_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        # Save representations
        representations_path = os.path.join(self.output_dir, f"{layer_name}_representations.pkl")
        with open(representations_path, 'wb') as f:
            pickle.dump(representations, f)

        # Analyze and visualize spatial relations
        if 'relation' in representations:
            # Analyze opposite relations
            relations_df = self.analyze_spatial_relations(representations)
            relations_df.to_csv(os.path.join(self.output_dir, f"{layer_name}_relation_analysis.csv"), index=False)

            # Visualize relation geometry
            for method in ['pca', 'tsne']:
                plt = self.visualize_relation_geometry(representations, method=method)
                plt.savefig(os.path.join(self.output_dir, f"{layer_name}_relation_{method}.png"), dpi=300, bbox_inches='tight')
                plt.close()

        return results, representations

    def run_all_layers(self, epochs=5):
        """Run analysis on all layers"""
        # Load data
        self.load_data()

        # Parse sentences
        self.parse_sentences()

        # Encode triplets
        self.encode_triplets()

        # Process each layer
        all_results = {}
        all_representations = {}

        for layer_name, layer_data in [
            ('layer8', self.layer8),
            ('layer16', self.layer16),
            ('layer24', self.layer24)
        ]:
            results, representations = self.process_layer(layer_name, layer_data, epochs=epochs)
            all_results[layer_name] = results
            all_representations[layer_name] = representations

        # Save all results
        with open(os.path.join(self.output_dir, "all_results.pkl"), 'wb') as f:
            pickle.dump(all_results, f)

        # Save all representations
        with open(os.path.join(self.output_dir, "all_representations.pkl"), 'wb') as f:
            pickle.dump(all_representations, f)

        # Create a summary report
        self.create_summary_report(all_results)

        return all_results, all_representations

    def create_summary_report(self, all_results):
        """Create a summary report of results across layers"""
        summary = []

        for layer_name in all_results:
            for component in all_results[layer_name]:
                summary.append({
                    'layer': layer_name,
                    'component': component,
                    'accuracy': all_results[layer_name][component]['accuracy']
                })

        summary_df = pd.DataFrame(summary)

        # Calculate average accuracy per layer
        layer_avg = summary_df.groupby('layer')['accuracy'].mean().reset_index()
        layer_avg['component'] = 'Average'

        # Combine with original data
        summary_df = pd.concat([summary_df, layer_avg])

        # Pivot for better visualization
        pivot_df = summary_df.pivot(index='component', columns='layer', values='accuracy')

        # Save to CSV
        pivot_df.to_csv(os.path.join(self.output_dir, "accuracy_summary.csv"))

        # Create plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title("Probe Accuracy by Layer and Component")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linear probe analysis for spatial world models")
    parser.add_argument("--hdf5", type=str, default="Llama-3.2-3B-Instruct_layer_embeddings.h5",
                        help="Path to HDF5 file with embeddings")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")

    args = parser.parse_args()

    # Initialize and run probe
    probe = SpatialLinearProbe(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=not args.cpu
    )

    results, representations = probe.run_all_layers(epochs=args.epochs)

    print("Analysis complete! Results saved to", args.output_dir)
