import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer

class EmailClassifierModel:
    """Email classifier model using embedding similarity"""

    def __init__(self):
        self.topic_data = self._load_topic_data()
        self.topics = list(self.topic_data.keys())

        # Load sentence transformer model (same model as feature generator)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute embeddings for all topic descriptions
        self.topic_embeddings = self._compute_topic_embeddings()
    
    @property
    def _data_file(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'topic_keywords.json')

    @property
    def _emails_file(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'emails.json')

    def _load_topic_data(self) -> Dict[str, Dict[str, Any]]:
        """Load topic data from data/topic_keywords.json"""
        with open(self._data_file, 'r') as f:
            return json.load(f)

    def create_topics(self, topics: List[Dict[str, str]]) -> str:
        """Persist one or more new topics to data/topic_keywords.json.
        Validates all entries before writing (all-or-nothing).
        """
        seen = set()
        for topic in topics:
            name = topic["name"]
            if name in self.topic_data:
                raise ValueError(f"Topic '{name}' already exists")
            if name in seen:
                raise ValueError(f"Duplicate topic '{name}' in request")
            seen.add(name)
        for topic in topics:
            self.topic_data[topic["name"]] = {"description": topic["description"]}
        with open(self._data_file, 'w') as f:
            json.dump(self.topic_data, f, indent=2)

    def _compute_topic_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all topic descriptions"""
        topic_embeddings = {}
        for topic, data in self.topic_data.items():
            description = data['description']
            embedding = self.model.encode(description, convert_to_numpy=True)
            topic_embeddings[topic] = embedding
        return topic_embeddings
    
    def predict(self, features: Dict[str, Any]) -> str:
        """Classify email into one of the topics using feature similarity"""
        scores = {}
        
        # Calculate similarity scores for each topic based on features
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = score
        
        return max(scores, key=scores.get)
    
    def get_topic_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get classification scores for all topics"""
        scores = {}
        
        for topic in self.topics:
            score = self._calculate_topic_score(features, topic)
            scores[topic] = float(score)
        
        return scores
    
    def _calculate_topic_score(self, features: Dict[str, Any], topic: str) -> float:
        """Calculate cosine similarity between email and topic embeddings"""
        # Get email embedding from features (now a list/array)
        email_embedding = features.get("email_embeddings_average_embedding", None)

        if email_embedding is None:
            return 0.0

        # Convert to numpy array if it's a list
        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        # Get pre-computed topic embedding
        topic_embedding = self.topic_embeddings[topic]

        # Calculate cosine similarity
        # cosine_similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(email_embedding, topic_embedding)
        email_norm = np.linalg.norm(email_embedding)
        topic_norm = np.linalg.norm(topic_embedding)

        if email_norm == 0 or topic_norm == 0:
            return 0.0

        cosine_similarity = dot_product / (email_norm * topic_norm)

        # Cosine similarity is between -1 and 1, but for text it's usually positive
        # Normalize to 0-1 range for better interpretability
        normalized_score = (cosine_similarity + 1) / 2

        return float(normalized_score)
    
    def predict_nearest_neighbor(self, features: Dict[str, Any]) -> Optional[str]:
        """Classify by finding the most similar labeled email in the stored emails file"""
        email_embedding = features.get("email_embeddings_average_embedding")
        if email_embedding is None:
            return None

        if isinstance(email_embedding, list):
            email_embedding = np.array(email_embedding)

        with open(self._emails_file, 'r') as f:
            stored_emails = json.load(f)

        labeled = [e for e in stored_emails if e.get("topic")]
        if not labeled:
            return None

        best_topic: Optional[str] = None
        best_score = -1.0

        for email in labeled:
            text = f"{email.get('subject', '')} {email.get('body', '')}".strip()
            stored_embedding = self.model.encode(text, convert_to_numpy=True)

            dot = np.dot(email_embedding, stored_embedding)
            norm_e = np.linalg.norm(email_embedding)
            norm_s = np.linalg.norm(stored_embedding)

            if norm_e == 0 or norm_s == 0:
                continue

            score = float(dot / (norm_e * norm_s))
            if score > best_score:
                best_score = score
                best_topic = email["topic"]

        return best_topic

    def get_topic_description(self, topic: str) -> str:
        """Get description for a specific topic"""
        return self.topic_data[topic]['description']
    
    def get_all_topics_with_descriptions(self) -> Dict[str, str]:
        """Get all topics with their descriptions"""
        return {topic: self.get_topic_description(topic) for topic in self.topics}