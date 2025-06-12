import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import json
import os
from datetime import datetime

# Import the comparative system
from comparative_fl_system import ComparativeFederatedLearningSystem

# Page configuration
st.set_page_config(
    page_title="Federated Learning Comparison System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'fl_system' not in st.session_state:
        st.session_state.fl_system = ComparativeFederatedLearningSystem()
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False

def main():
    init_session_state()
    
    st.title("🤖 Federated Learning Comparison System")
    st.markdown("### Compare Federated Learning with and without Differential Privacy")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Experiment Configuration")
        
        # Number of clients selection
        num_clients = st.slider(
            "Number of Clients", 
            min_value=2, 
            max_value=20, 
            value=5,
            help="Select the number of federated learning clients"
        )
        
        # Number of rounds
        num_rounds = st.slider(
            "Training Rounds", 
            min_value=5, 
            max_value=50, 
            value=10,
            help="Number of federated learning rounds"
        )
        
        st.subheader("🔐 Privacy Parameters")
        
        # Privacy budget (epsilon)
        epsilon = st.slider(
            "Privacy Budget (ε)", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Lower values = more privacy but potentially lower accuracy"
        )
        
        # Delta parameter
        delta = st.select_slider(
            "Delta (δ)", 
            options=[1e-6, 1e-5, 1e-4, 1e-3], 
            value=1e-5,
            format_func=lambda x: f"{x:.0e}",
            help="Probability of privacy breach"
        )
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Experiment", type="primary", disabled=st.session_state.experiment_running):
                st.session_state.experiment_running = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.fl_system.clear_results()
                st.rerun()
    
    # Main content area
    if st.session_state.experiment_running:
        st.header("🔄 Running Experiment...")
        
        # Run the comparison experiment
        try:
            result = st.session_state.fl_system.run_comparison_experiment(
                num_clients=num_clients,
                epsilon=epsilon,
                delta=delta,
                num_rounds=num_rounds
            )
            
            st.success(f"✅ Experiment completed with {num_clients} clients!")
            
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
            
            st.session_state.experiment_running = False
            
        except Exception as e:
            st.error(f"Experiment failed: {str(e)}")
            st.session_state.experiment_running = False
    
    # Display results and visualizations
    st.header("📊 Results Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Visualizations", "📋 Results Table", "💾 Data Export"])
    
    with tab1:
        st.session_state.fl_system.create_comparison_visualizations()
    
    with tab2:
        st.subheader("Experiment Results")
        st.session_state.fl_system.display_results_table()
    
    with tab3:
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
    1. **Select Parameters**: Choose the number of clients and privacy settings in the sidebar
    2. **Run Experiment**: Click "Run Experiment" to execute both privacy modes
    3. **View Results**: Analyze the comparison charts and detailed results
    4. **Repeat**: Try different client numbers to build a comprehensive comparison
    5. **Export**: Download your results for further analysis
    """)

if __name__ == "__main__":
    main()