import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import threading
import math
import json
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Import custom modules
from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from data_distribution import get_distribution_strategy, visualize_data_distribution
from fog_aggregation import HierarchicalFederatedLearning
from differential_privacy import DifferentialPrivacyManager
from hierarchical_fl_protocol import HierarchicalFederatedLearningEngine
from client_visualization import ClientPerformanceVisualizer
from journey_visualization import InteractiveJourneyVisualizer
from performance_optimizer import create_performance_optimizer
from advanced_client_analytics import AdvancedClientAnalytics
from real_medical_data_fetcher import RealMedicalDataFetcher, load_authentic_medical_data
from training_secret_sharing import TrainingLevelSecretSharingManager, integrate_training_secret_sharing
from translations import get_translation, translate_risk_level, translate_clinical_advice
from comparative_fl_system import ComparativeFederatedLearningSystem
from client_simulator import ClientSimulator

from utils import *

# Page configuration
st.set_page_config(
    page_title=get_translation("page_title"),
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []
    if 'fog_results' not in st.session_state:
        st.session_state.fog_results = []
    if 'client_results' not in st.session_state:
        st.session_state.client_results = []
    if 'execution_times' not in st.session_state:
        st.session_state.execution_times = []
    if 'communication_times' not in st.session_state:
        st.session_state.communication_times = []
    if 'confusion_matrices' not in st.session_state:
        st.session_state.confusion_matrices = []
    if 'early_stopped' not in st.session_state:
        st.session_state.early_stopped = False
    if 'best_accuracy' not in st.session_state:
        st.session_state.best_accuracy = 0.0
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'current_training_round' not in st.session_state:
        st.session_state.current_training_round = 0
    if 'round_client_metrics' not in st.session_state:
        st.session_state.round_client_metrics = {}
    if 'client_progress' not in st.session_state:
        st.session_state.client_progress = {}
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'client_visualizer' not in st.session_state:
        st.session_state.client_visualizer = ClientPerformanceVisualizer()
    if 'journey_visualizer' not in st.session_state:
        st.session_state.journey_visualizer = InteractiveJourneyVisualizer()
    if 'advanced_analytics' not in st.session_state:
        st.session_state.advanced_analytics = AdvancedClientAnalytics()
    if 'fl_system' not in st.session_state:
        st.session_state.fl_system = ComparativeFederatedLearningSystem()
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False


def main():
    init_session_state()
    
    st.title(get_translation("page_title", st.session_state.language))
    st.markdown("### " + get_translation("advanced_privacy_preserving_ml_platform", st.session_state.language))

    # Sidebar
    with st.sidebar:
        # Language selector at top
        st.markdown("### 🌐 Language / Langue")
        selected_language = st.selectbox(
            get_translation("language_selector", st.session_state.language),
            options=["English", "Français"],
            index=0 if st.session_state.language == 'en' else 1,
            key="language_selector"
        )
        
        # Update language in session state
        if selected_language == "English":
            st.session_state.language = 'en'
        else:
            st.session_state.language = 'fr'
        
        st.markdown("---")
        st.header("🔧 System Configuration")
        
        # Data upload
        uploaded_file = st.file_uploader("📁 Upload Patient Dataset", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.success(f"✅ Dataset loaded: {data.shape[0]} patients, {data.shape[1]} features")
        
        # Always ensure data is loaded
        if not hasattr(st.session_state, 'data') or st.session_state.data is None:
            try:
                data = pd.read_csv('diabetes.csv')
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.success(f"📊 Diabetes dataset loaded: {data.shape[0]} patients, {data.shape[1]} features")
            except Exception as e:
                st.error(f"Failed to load diabetes dataset: {str(e)}")
                return

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        get_translation("tab_training", st.session_state.language), 
        get_translation("tab_monitoring", st.session_state.language), 
        get_translation("tab_visualization", st.session_state.language),
        get_translation("tab_analytics", st.session_state.language),
        get_translation("tab_facility", st.session_state.language),
        get_translation("tab_risk", st.session_state.language),
        get_translation("tab_graph_viz", st.session_state.language),
        "📊 Advanced Analytics",
        "🔬 FL Comparison"
    ])

    with tab1:
        st.header("🎛️ " + get_translation("tab_training"))
        
        if st.session_state.data_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(get_translation("medical_network_config", st.session_state.language))
                # Use session state to control default values for reset functionality
                if 'reset_requested' in st.session_state and st.session_state.reset_requested:
                    default_clients = 5
                    default_rounds = 20
                    st.session_state.reset_requested = False
                else:
                    default_clients = st.session_state.get('num_clients', 5)
                    default_rounds = st.session_state.get('max_rounds', 20)
                
                num_clients = st.slider(get_translation("num_medical_stations", st.session_state.language), 3, 20, default_clients)
                max_rounds = st.slider(get_translation("max_training_rounds", st.session_state.language), 10, 150, default_rounds)
                
                # Early stopping configuration
                st.subheader("🛑 Early Stopping Configuration")
                
                col1_es, col2_es = st.columns(2)
                
                with col1_es:
                    enable_early_stopping = st.checkbox("Enable Early Stopping", value=True,
                                                       help="Stop training when validation metric stops improving")
                    
                    patience = st.slider("Patience (rounds)", min_value=3, max_value=20, value=5,
                                       help="Number of rounds to wait without improvement before stopping")
                
                with col2_es:
                    early_stop_metric = st.selectbox("Early Stop Metric", 
                                                    ["accuracy", "loss", "f1_score"], 
                                                    index=0,
                                                    help="Metric to monitor for early stopping")
                    
                    min_improvement = st.number_input("Minimum Improvement", 
                                                    min_value=0.001, max_value=0.1, 
                                                    value=0.001, step=0.001, format="%.3f",
                                                    help="Minimum improvement required to reset patience counter")
                
                if enable_early_stopping:
                    st.info(f"Training will stop if {early_stop_metric} doesn't improve by {min_improvement:.3f} for {patience} consecutive rounds")
                else:
                    st.warning("Early stopping disabled - training will run for full duration")
                
                # Store values in session state
                st.session_state.num_clients = num_clients
                st.session_state.max_rounds = max_rounds
                st.session_state.enable_early_stopping = enable_early_stopping
                st.session_state.patience = patience
                st.session_state.early_stop_metric = early_stop_metric
                st.session_state.min_improvement = min_improvement
                
                st.subheader(get_translation("model_selection", st.session_state.language))
                default_model = "Deep Learning (Neural Network)" if 'reset_requested' in st.session_state else st.session_state.get('model_type_display', "Deep Learning (Neural Network)")
                model_type = st.selectbox(get_translation("machine_learning_model", st.session_state.language), 
                                        ["Deep Learning (Neural Network)", "CNN (Convolutional)", "SVM (Support Vector)", "Logistic Regression", "Random Forest"],
                                        index=["Deep Learning (Neural Network)", "CNN (Convolutional)", "SVM (Support Vector)", "Logistic Regression", "Random Forest"].index(default_model),
                                        help="Select the AI model type for diabetes prediction")
                st.session_state.model_type_display = model_type
                
                # Map display names to internal names
                model_mapping = {
                    "Deep Learning (Neural Network)": "neural_network",
                    "CNN (Convolutional)": "cnn", 
                    "SVM (Support Vector)": "svm",
                    "Logistic Regression": "logistic_regression",
                    "Random Forest": "random_forest"
                }
                internal_model_type = model_mapping[model_type]
                
                st.subheader(get_translation("fog_computing_setup", st.session_state.language))
                default_fog = True if 'reset_requested' not in st.session_state else True
                enable_fog = st.checkbox(get_translation("enable_fog_nodes", st.session_state.language), value=st.session_state.get('enable_fog', default_fog))
                st.session_state.enable_fog = enable_fog
                
                if enable_fog:
                    default_fog_nodes = 3 if 'reset_requested' not in st.session_state else 3
                    num_fog_nodes = st.slider(get_translation("num_fog_nodes", st.session_state.language), 2, 6, st.session_state.get('num_fog_nodes', default_fog_nodes))
                    st.session_state.num_fog_nodes = num_fog_nodes
                    
                    default_fog_method = "FedAvg" if 'reset_requested' not in st.session_state else "FedAvg"
                    fog_methods = ["FedAvg", "FedProx", "Weighted", "Median", "Mixed Methods"]
                    current_method = st.session_state.get('fog_method', default_fog_method)
                    fog_method = st.selectbox(get_translation("fog_aggregation_method", st.session_state.language), fog_methods, index=fog_methods.index(current_method))
                    st.session_state.fog_method = fog_method
                else:
                    num_fog_nodes = 0
                    fog_method = "FedAvg"
            
            with col2:
                st.subheader(get_translation("privacy_configuration", st.session_state.language))
                enable_dp = st.checkbox(get_translation("enable_privacy", st.session_state.language), value=True, key="enable_dp_check")
                if enable_dp:
                    epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1, key="epsilon_slider")
                    delta = st.select_slider("Failure Probability (δ)", 
                                           options=[1e-6, 1e-5, 1e-4, 1e-3], 
                                           value=1e-5, format_func=lambda x: f"{x:.0e}", key="delta_select")
                    
                    # Store in session state for federated learning manager to access
                    st.session_state.epsilon = epsilon
                    st.session_state.delta = delta
                else:
                    epsilon = None
                    delta = None
                    st.session_state.epsilon = None
                    st.session_state.delta = None
                
                st.subheader(get_translation("data_distribution", st.session_state.language))
                distribution_strategy = st.selectbox(get_translation("distribution_strategy", st.session_state.language), 
                                                   ["IID", "Non-IID", "Pathological", "Quantity Skew", "Geographic"], key="distribution_select")
                
                # Strategy-specific parameters
                strategy_params = {}
                if distribution_strategy == "Non-IID":
                    strategy_params['alpha'] = st.slider("Dirichlet Alpha", 0.1, 2.0, 0.5, 0.1, key="alpha_slider")
                elif distribution_strategy == "Pathological":
                    strategy_params['classes_per_client'] = st.slider("Classes per Client", 1, 2, 1, key="classes_slider")
                elif distribution_strategy == "Quantity Skew":
                    strategy_params['skew_factor'] = st.slider("Skew Factor", 1.0, 5.0, 2.0, 0.5, key="skew_slider")
                
                if distribution_strategy == "Geographic":
                    strategy_params['correlation_strength'] = st.slider("Correlation Strength", 0.1, 1.0, 0.8, 0.1, key="correlation_slider")
                
                st.subheader("🔐 Training-Level Secret Sharing")
                enable_training_ss = st.checkbox("Enable Secret Sharing in Training", value=True, key="enable_ss_check")
                if enable_training_ss:
                    if enable_fog:
                        ss_threshold = st.slider("Secret Sharing Threshold", 
                                               min_value=2, 
                                               max_value=num_fog_nodes, 
                                               value=max(2, int(0.67 * num_fog_nodes)),
                                               help=f"Number of fog nodes required to reconstruct weights (max: {num_fog_nodes})",
                                               key="ss_threshold_slider")
                        st.info(f"Using {num_fog_nodes} fog nodes for secret sharing distribution")
                        st.success(f"Secret sharing: {ss_threshold}/{num_fog_nodes} threshold scheme")
                    else:
                        st.warning("Enable Fog Nodes to use secret sharing")
                        enable_training_ss = False
                        ss_threshold = 3  # Default value when disabled
                else:
                    ss_threshold = 3  # Default value when disabled
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start Federated Training", type="primary", disabled=st.session_state.training_in_progress):
                    # Store configuration
                    config = {
                        'num_clients': num_clients,
                        'max_rounds': max_rounds,
                        'model_type': internal_model_type,
                        'enable_dp': enable_dp,
                        'epsilon': epsilon,
                        'delta': delta,
                        'distribution_strategy': distribution_strategy,
                        'strategy_params': strategy_params,
                        'enable_fog': enable_fog,
                        'num_fog_nodes': num_fog_nodes,
                        'fog_method': fog_method,
                        'enable_training_ss': enable_training_ss,
                        'ss_threshold': ss_threshold,
                        'enable_early_stopping': enable_early_stopping,
                        'patience': patience,
                        'early_stop_metric': early_stop_metric,
                        'min_improvement': min_improvement
                    }
                    
                    st.session_state.training_config = config
                    st.session_state.training_started = True
                    st.session_state.training_in_progress = True
                    st.session_state.current_round = 0
                    st.session_state.training_metrics = []
                    st.session_state.client_results = []
                    
                    # Initialize federated learning manager
                    fl_manager = FederatedLearningManager(
                        num_clients=num_clients,
                        max_rounds=max_rounds,
                        enable_dp=enable_dp,
                        epsilon=epsilon or 1.0,
                        delta=delta or 1e-5,
                        model_type=internal_model_type,
                        enable_early_stopping=enable_early_stopping,
                        patience=patience,
                        early_stop_metric=early_stop_metric,
                        min_improvement=min_improvement
                    )
                    st.session_state.fl_manager = fl_manager
                    
                    st.rerun()
            
            with col2:
                if st.button("⏹️ Stop Training", disabled=not st.session_state.training_in_progress):
                    st.session_state.training_in_progress = False
                    st.session_state.training_started = False
                    st.warning("Training stopped by user")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset System"):
                    # Clear all session state variables related to training
                    keys_to_clear = [
                        'training_started', 'training_completed', 'training_metrics',
                        'fog_results', 'client_results', 'execution_times',
                        'communication_times', 'confusion_matrices', 'early_stopped',
                        'best_accuracy', 'current_round', 'results',
                        'current_training_round', 'round_client_metrics',
                        'client_progress', 'training_in_progress'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Reset to defaults
                    st.session_state.reset_requested = True
                    st.success("System reset complete!")
                    st.rerun()

        else:
            st.warning(get_translation("please_upload_data", st.session_state.language))

    with tab9:  # FL Comparison tab
        st.header("🔬 Federated Learning Comparison System")
        st.markdown("### Compare Federated Learning with and without Differential Privacy")
        
        # Configuration section
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 Experiment Configuration")
                
                # Number of clients selection
                comp_num_clients = st.slider(
                    "Number of Clients", 
                    min_value=2, 
                    max_value=20, 
                    value=5,
                    help="Select the number of federated learning clients",
                    key="comp_clients"
                )
                
                # Number of rounds
                comp_num_rounds = st.slider(
                    "Training Rounds", 
                    min_value=5, 
                    max_value=50, 
                    value=10,
                    help="Number of federated learning rounds",
                    key="comp_rounds"
                )
                
            with col2:
                st.subheader("🔐 Privacy Parameters")
                
                # Privacy budget (epsilon)
                comp_epsilon = st.slider(
                    "Privacy Budget (ε)", 
                    min_value=0.1, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1,
                    help="Lower values = more privacy but potentially lower accuracy",
                    key="comp_epsilon"
                )
                
                # Delta parameter
                comp_delta = st.select_slider(
                    "Delta (δ)", 
                    options=[1e-6, 1e-5, 1e-4, 1e-3], 
                    value=1e-5,
                    format_func=lambda x: f"{x:.0e}",
                    help="Probability of privacy breach",
                    key="comp_delta"
                )
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Comparison Experiment", type="primary", disabled=st.session_state.experiment_running):
                st.session_state.experiment_running = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Comparison Results"):
                st.session_state.fl_system.clear_results()
                st.rerun()
        
        # Run experiment if requested
        if st.session_state.experiment_running:
            st.header("🔄 Running Comparison Experiment...")
            
            # Run the comparison experiment
            try:
                result = st.session_state.fl_system.run_comparison_experiment(
                    num_clients=comp_num_clients,
                    epsilon=comp_epsilon,
                    delta=comp_delta,
                    num_rounds=comp_num_rounds
                )
                
                if result:
                    st.success(f"✅ Experiment completed with {comp_num_clients} clients!")
                    
                    # Show immediate results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Accuracy without DP", 
                            f"{result['without_dp']['final_accuracy']:.4f}",
                            f"{result['without_dp']['final_accuracy']:.4f}"
                        )
                        st.metric(
                            "Runtime without DP", 
                            f"{result['without_dp']['runtime']:.2f}s"
                        )
                    
                    with col2:
                        st.metric(
                            "Accuracy with DP", 
                            f"{result['with_dp']['final_accuracy']:.4f}",
                            f"{result['with_dp']['final_accuracy'] - result['without_dp']['final_accuracy']:.4f}"
                        )
                        st.metric(
                            "Runtime with DP", 
                            f"{result['with_dp']['runtime']:.2f}s"
                        )
                else:
                    st.error("Failed to load dataset for comparison")
                
                st.session_state.experiment_running = False
                
            except Exception as e:
                st.error(f"Experiment failed: {str(e)}")
                st.session_state.experiment_running = False
        
        # Display results and visualizations
        st.header("📊 Comparison Results Analysis")
        
        # Create tabs for different views
        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📈 Visualizations", "📋 Results Table", "💾 Data Export"])
        
        with comp_tab1:
            st.session_state.fl_system.create_comparison_visualizations()
        
        with comp_tab2:
            st.subheader("Experiment Results")
            st.session_state.fl_system.display_results_table()
        
        with comp_tab3:
            st.subheader("Export Results")
            
            if st.session_state.fl_system.results_data:
                # Convert results to DataFrame for export
                export_data = []
                for result in st.session_state.fl_system.results_data:
                    export_data.append({
                        'timestamp': result['timestamp'],
                        'num_clients': result['num_clients'],
                        'num_rounds': result['num_rounds'],
                        'epsilon': result.get('epsilon', 1.0),
                        'delta': result.get('delta', 1e-5),
                        'accuracy_no_dp': result['without_dp']['final_accuracy'],
                        'accuracy_with_dp': result['with_dp']['final_accuracy'],
                        'runtime_no_dp': result['without_dp']['runtime'],
                        'runtime_with_dp': result['with_dp']['runtime'],
                        'loss_no_dp': result['without_dp']['final_loss'],
                        'loss_with_dp': result['with_dp']['final_loss'],
                        'f1_no_dp': result['without_dp']['final_f1'],
                        'f1_with_dp': result['with_dp']['final_f1']
                    })
                
                export_df = pd.DataFrame(export_data)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"fl_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = json.dumps(st.session_state.fl_system.results_data, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"fl_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("No results available for export. Run some experiments first!")
        
        # Footer with instructions
        st.markdown("---")
        st.markdown("""
        ### 📋 How to Use:
        1. **Select Parameters**: Choose the number of clients and privacy settings
        2. **Run Experiment**: Click "Run Comparison Experiment" to execute both privacy modes
        3. **View Results**: Analyze the comparison charts and detailed results
        4. **Repeat**: Try different client numbers to build a comprehensive comparison
        5. **Export**: Download your results for further analysis
        """)

    with tab2:
        st.header("📊 " + get_translation("tab_monitoring"))
        
        if not st.session_state.training_started:
            st.info("Configure training in the Training tab first")
        else:
            # Display training progress and real-time monitoring
            if hasattr(st.session_state, 'fl_manager') and st.session_state.training_in_progress:
                
                # Training status section
                st.subheader("🔄 Training Status")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Round", st.session_state.current_round)
                with col2:
                    st.metric("Max Rounds", st.session_state.training_config['max_rounds'])
                with col3:
                    st.metric("Active Clients", st.session_state.training_config['num_clients'])
                with col4:
                    if st.session_state.training_metrics:
                        latest_accuracy = st.session_state.training_metrics[-1].get('accuracy', 0)
                        st.metric("Latest Accuracy", f"{latest_accuracy:.3f}")
                    else:
                        st.metric("Latest Accuracy", "N/A")
                
                # Progress bar
                progress = st.session_state.current_round / st.session_state.training_config['max_rounds']
                st.progress(progress)
                
                # Execute training step if in progress
                if st.session_state.training_in_progress and st.session_state.current_round < st.session_state.training_config['max_rounds']:
                    
                    # Run one training round
                    try:
                        # Get training data
                        data = st.session_state.data
                        preprocessor = DataPreprocessor()
                        
                        # Prepare data
                        X = data.drop('Outcome', axis=1).values.astype(np.float32)
                        y = data['Outcome'].values.astype(np.int32)
                        
                        # Split for training
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Create data distribution
                        distribution_strategy = get_distribution_strategy(
                            st.session_state.training_config['distribution_strategy'], 
                            st.session_state.training_config['num_clients'],
                            **st.session_state.training_config['strategy_params']
                        )
                        
                        client_data = distribution_strategy.distribute_data(X_train, y_train)
                        
                        # Initialize clients if not already done
                        if not hasattr(st.session_state, 'clients'):
                            clients = []
                            for i, client_data_dict in enumerate(client_data):
                                client = ClientSimulator(
                                    client_id=i,
                                    data=client_data_dict,
                                    model_type=st.session_state.training_config['model_type']
                                )
                                clients.append(client)
                            st.session_state.clients = clients
                        
                        # Run training round
                        round_metrics = {}
                        client_accuracies = []
                        
                        # Train each client
                        for client in st.session_state.clients:
                            client_result = client.train(local_epochs=1)
                            client_eval = client.evaluate()
                            
                            if client_eval:
                                client_accuracies.append(client_eval.get('test_accuracy', 0.5))
                        
                        # Calculate round metrics
                        round_metrics = {
                            'round': st.session_state.current_round + 1,
                            'accuracy': np.mean(client_accuracies) if client_accuracies else 0.5,
                            'std_accuracy': np.std(client_accuracies) if client_accuracies else 0.0,
                            'num_clients': len(st.session_state.clients),
                            'timestamp': time.time()
                        }
                        
                        # Store metrics
                        st.session_state.training_metrics.append(round_metrics)
                        st.session_state.current_round += 1
                        
                        # Check early stopping
                        if st.session_state.training_config['enable_early_stopping']:
                            current_metric = round_metrics['accuracy']
                            if not hasattr(st.session_state, 'best_metric') or current_metric > st.session_state.best_metric:
                                st.session_state.best_metric = current_metric
                                st.session_state.patience_counter = 0
                            else:
                                st.session_state.patience_counter += 1
                                
                            if st.session_state.patience_counter >= st.session_state.training_config['patience']:
                                st.session_state.training_in_progress = False
                                st.session_state.early_stopped = True
                                st.success(f"Early stopping triggered at round {st.session_state.current_round}")
                        
                        # Check if training complete
                        if st.session_state.current_round >= st.session_state.training_config['max_rounds']:
                            st.session_state.training_in_progress = False
                            st.session_state.training_completed = True
                            st.success("Training completed!")
                        
                        # Auto-refresh for next round
                        if st.session_state.training_in_progress:
                            time.sleep(1)  # Brief pause between rounds
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        st.session_state.training_in_progress = False
                
                # Real-time metrics display
                if st.session_state.training_metrics:
                    st.subheader("📈 Training Progress")
                    
                    # Create metrics DataFrame
                    metrics_df = pd.DataFrame(st.session_state.training_metrics)
                    
                    # Plot accuracy over rounds
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=metrics_df['round'],
                        y=metrics_df['accuracy'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue', width=3)
                    ))
                    
                    fig.update_layout(
                        title='Training Accuracy Over Rounds',
                        xaxis_title='Round',
                        yaxis_title='Accuracy',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display recent metrics table
                    st.subheader("📋 Recent Training Metrics")
                    recent_metrics = metrics_df.tail(10)
                    st.dataframe(recent_metrics, use_container_width=True)
            
            elif st.session_state.training_completed:
                st.success("✅ Training Completed Successfully!")
                
                # Display final results
                if st.session_state.training_metrics:
                    final_metrics = st.session_state.training_metrics[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Accuracy", f"{final_metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Total Rounds", final_metrics['round'])
                    with col3:
                        early_stop_text = "Yes" if getattr(st.session_state, 'early_stopped', False) else "No"
                        st.metric("Early Stopped", early_stop_text)
            
            else:
                st.info("Training configured but not yet started. Click 'Start Federated Training' in the Training tab.")
    
    with tab3:
        st.header("📈 " + get_translation("tab_visualization"))
        st.info("Visualization interface - Run training to see results")
    
    with tab4:
        st.header("📊 " + get_translation("tab_analytics"))
        st.info("Analytics interface - Results will appear after training")
    
    with tab5:
        st.header("🏥 " + get_translation("tab_facility"))
        st.info("Facility interface - Medical facility analytics")
    
    with tab6:
        st.header("⚠️ " + get_translation("tab_risk"))
        st.info("Risk assessment interface")
    
    with tab7:
        st.header("🔗 " + get_translation("tab_graph_viz"))
        st.info("Graph visualization interface")
    
    with tab8:
        st.header("📊 Advanced Analytics")
        st.info("Advanced analytics interface")

if __name__ == "__main__":
    main()