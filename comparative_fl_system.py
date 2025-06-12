import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from data_distribution import get_distribution_strategy
from differential_privacy import DifferentialPrivacyManager
from client_simulator import ClientSimulator
from aggregation_algorithms import FedAvgAggregator

class ComparativeFederatedLearningSystem:
    """System for comparing federated learning with and without differential privacy"""
    
    def __init__(self):
        self.results_file = "fl_comparison_results.json"
        self.results_data = self.load_results()
        
    def load_results(self):
        """Load existing results from file"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_results(self, result_entry):
        """Save results to file"""
        self.results_data.append(result_entry)
        with open(self.results_file, 'w') as f:
            json.dump(self.results_data, f, indent=2, default=str)
    
    def load_diabetes_dataset(self):
        """Load the real diabetes dataset"""
        try:
            df = pd.read_csv('diabetes.csv')
            return df
        except FileNotFoundError:
            st.error("Diabetes dataset not found. Please ensure diabetes.csv is in the current directory.")
            return None
    
    def run_federated_learning(self, data, num_clients, use_differential_privacy=False, 
                             epsilon=1.0, delta=1e-5, num_rounds=10):
        """Run federated learning experiment"""
        start_time = time.time()
        
        # Prepare data
        X = data.drop('Outcome', axis=1).values
        y = data['Outcome'].values
        
        # Split data for federated learning
        distribution_strategy = get_distribution_strategy("IID", num_clients)
        client_data = distribution_strategy.distribute_data(X, y)
        
        # Initialize privacy manager if needed
        privacy_manager = None
        if use_differential_privacy:
            privacy_manager = DifferentialPrivacyManager(epsilon=epsilon, delta=delta)
        
        # Initialize aggregator
        aggregator = FedAvgAggregator()
        
        # Initialize global model
        global_model = LogisticRegression(random_state=42, max_iter=1000)
        global_model.fit(X[:100], y[:100])  # Initialize with small sample
        
        # Initialize clients
        clients = []
        for i, client_data_dict in enumerate(client_data):
            client = ClientSimulator(
                client_id=i,
                data=client_data_dict,
                model_type='logistic_regression'
            )
            clients.append(client)
        
        # Training metrics
        round_metrics = []
        
        # Federated learning rounds
        for round_num in range(num_rounds):
            client_updates = []
            round_accuracies = []
            
            # Client training
            for client in clients:
                client.receive_global_model(global_model)
                update = client.train(local_epochs=1)
                
                if use_differential_privacy and privacy_manager:
                    # Add noise to client update
                    noisy_updates = privacy_manager.add_noise([update])
                    update = noisy_updates[0]
                
                client_updates.append(update)
                
                # Evaluate client
                client_eval = client.evaluate()
                if client_eval and 'test_accuracy' in client_eval:
                    client_accuracy = client_eval['test_accuracy']
                else:
                    client_accuracy = 0.5  # Default fallback
                round_accuracies.append(client_accuracy)
            
            # Aggregate updates
            global_model = aggregator.aggregate(global_model, client_updates)
            
            # Global evaluation
            global_predictions = global_model.predict(X)
            global_accuracy = accuracy_score(y, global_predictions)
            global_precision = precision_score(y, global_predictions, average='weighted')
            global_recall = recall_score(y, global_predictions, average='weighted')
            global_f1 = f1_score(y, global_predictions, average='weighted')
            
            try:
                global_proba = global_model.predict_proba(X)
                global_loss = log_loss(y, global_proba)
            except:
                global_loss = 0.0
            
            round_metrics.append({
                'round': round_num + 1,
                'global_accuracy': global_accuracy,
                'global_precision': global_precision,
                'global_recall': global_recall,
                'global_f1': global_f1,
                'global_loss': global_loss,
                'avg_client_accuracy': np.mean(round_accuracies),
                'std_client_accuracy': np.std(round_accuracies)
            })
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Final metrics
        final_metrics = round_metrics[-1] if round_metrics else {}
        
        return {
            'runtime': runtime,
            'final_accuracy': final_metrics.get('global_accuracy', 0),
            'final_precision': final_metrics.get('global_precision', 0),
            'final_recall': final_metrics.get('global_recall', 0),
            'final_f1': final_metrics.get('global_f1', 0),
            'final_loss': final_metrics.get('global_loss', 0),
            'avg_client_accuracy': final_metrics.get('avg_client_accuracy', 0),
            'round_metrics': round_metrics
        }
    
    def run_comparison_experiment(self, num_clients, epsilon=1.0, delta=1e-5, num_rounds=10):
        """Run both with and without differential privacy"""
        
        # Load real diabetes dataset
        data = self.load_diabetes_dataset()
        if data is None:
            return None
        
        st.info(f"Running experiment with {num_clients} clients...")
        
        # Progress bars
        progress_container = st.container()
        with progress_container:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Without Differential Privacy")
                progress_bar_1 = st.progress(0)
                status_1 = st.empty()
                
            with col2:
                st.subheader("With Differential Privacy")
                progress_bar_2 = st.progress(0)
                status_2 = st.empty()
        
        # Run without differential privacy
        status_1.text("Training without privacy...")
        progress_bar_1.progress(50)
        
        results_no_dp = self.run_federated_learning(
            data, num_clients, use_differential_privacy=False, num_rounds=num_rounds
        )
        
        progress_bar_1.progress(100)
        status_1.text("✅ Completed without privacy")
        
        # Run with differential privacy
        status_2.text("Training with privacy...")
        progress_bar_2.progress(50)
        
        results_with_dp = self.run_federated_learning(
            data, num_clients, use_differential_privacy=True, 
            epsilon=epsilon, delta=delta, num_rounds=num_rounds
        )
        
        progress_bar_2.progress(100)
        status_2.text("✅ Completed with privacy")
        
        # Create result entry
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'epsilon': epsilon,
            'delta': delta,
            'without_dp': results_no_dp,
            'with_dp': results_with_dp
        }
        
        # Save results
        self.save_results(result_entry)
        
        return result_entry
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        if not self.results_data:
            st.warning("No results data available. Please run some experiments first.")
            return
        
        # Prepare data for visualization
        comparison_data = []
        for result in self.results_data:
            comparison_data.append({
                'num_clients': result['num_clients'],
                'accuracy_no_dp': result['without_dp']['final_accuracy'],
                'accuracy_with_dp': result['with_dp']['final_accuracy'],
                'runtime_no_dp': result['without_dp']['runtime'],
                'runtime_with_dp': result['with_dp']['runtime'],
                'loss_no_dp': result['without_dp']['final_loss'],
                'loss_with_dp': result['with_dp']['final_loss'],
                'f1_no_dp': result['without_dp']['final_f1'],
                'f1_with_dp': result['with_dp']['final_f1'],
                'epsilon': result.get('epsilon', 1.0),
                'timestamp': result['timestamp']
            })
        
        df = pd.DataFrame(comparison_data)
        
        if df.empty:
            st.warning("No valid comparison data available.")
            return
        
        # Sort by number of clients
        df = df.sort_values('num_clients')
        
        st.subheader("📊 Comprehensive Comparison Results")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Accuracy Comparison", "Runtime Analysis", "Loss Analysis", "F1 Score Analysis"])
        
        with tab1:
            self._create_accuracy_comparison(df)
        
        with tab2:
            self._create_runtime_comparison(df)
            
        with tab3:
            self._create_loss_comparison(df)
            
        with tab4:
            self._create_f1_comparison(df)
    
    def _create_accuracy_comparison(self, df):
        """Create accuracy comparison visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['accuracy_no_dp'],
            mode='lines+markers',
            name='Without Differential Privacy',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['accuracy_with_dp'],
            mode='lines+markers',
            name='With Differential Privacy',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Accuracy Comparison: With vs Without Differential Privacy',
            xaxis_title='Number of Clients',
            yaxis_title='Accuracy',
            legend=dict(x=0.7, y=0.95),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show accuracy difference
        if len(df) > 1:
            df['accuracy_difference'] = df['accuracy_no_dp'] - df['accuracy_with_dp']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df['num_clients'],
                y=df['accuracy_difference'],
                name='Accuracy Loss due to Privacy',
                marker_color='orange'
            ))
            
            fig2.update_layout(
                title='Accuracy Loss Due to Differential Privacy',
                xaxis_title='Number of Clients',
                yaxis_title='Accuracy Difference',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def _create_runtime_comparison(self, df):
        """Create runtime comparison visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['runtime_no_dp'],
            mode='lines+markers',
            name='Without DP',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['runtime_with_dp'],
            mode='lines+markers',
            name='With DP',
            line=dict(color='purple', width=3)
        ))
        
        fig.update_layout(
            title='Runtime Comparison',
            xaxis_title='Number of Clients',
            yaxis_title='Runtime (seconds)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_loss_comparison(self, df):
        """Create loss comparison visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['loss_no_dp'],
            mode='lines+markers',
            name='Loss without DP',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['loss_with_dp'],
            mode='lines+markers',
            name='Loss with DP',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Loss Comparison',
            xaxis_title='Number of Clients',
            yaxis_title='Loss',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_f1_comparison(self, df):
        """Create F1 score comparison visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['f1_no_dp'],
            mode='lines+markers',
            name='F1 without DP',
            line=dict(color='cyan', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['num_clients'],
            y=df['f1_with_dp'],
            mode='lines+markers',
            name='F1 with DP',
            line=dict(color='magenta', width=3)
        ))
        
        fig.update_layout(
            title='F1 Score Comparison',
            xaxis_title='Number of Clients',
            yaxis_title='F1 Score',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_results_table(self):
        """Display results in a comprehensive table"""
        if not self.results_data:
            st.warning("No results available yet.")
            return
        
        # Prepare table data
        table_data = []
        for result in self.results_data:
            table_data.append({
                'Timestamp': result['timestamp'][:19],
                'Clients': result['num_clients'],
                'Rounds': result['num_rounds'],
                'Epsilon': result.get('epsilon', 1.0),
                'Accuracy (No DP)': f"{result['without_dp']['final_accuracy']:.4f}",
                'Accuracy (With DP)': f"{result['with_dp']['final_accuracy']:.4f}",
                'Runtime (No DP)': f"{result['without_dp']['runtime']:.2f}s",
                'Runtime (With DP)': f"{result['with_dp']['runtime']:.2f}s",
                'Loss (No DP)': f"{result['without_dp']['final_loss']:.4f}",
                'Loss (With DP)': f"{result['with_dp']['final_loss']:.4f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    def clear_results(self):
        """Clear all stored results"""
        self.results_data = []
        if os.path.exists(self.results_file):
            os.remove(self.results_file)
        st.success("All results cleared!")