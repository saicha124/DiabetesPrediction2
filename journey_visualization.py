import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Tuple

class InteractiveJourneyVisualizer:
    """Interactive visualization of the federated learning user journey"""
    
    def __init__(self):
        self.journey_stages = [
            "Data Loading",
            "Configuration", 
            "Client Setup",
            "Privacy Setup",
            "Training Initiation",
            "Fog Aggregation",
            "Global Convergence",
            "Model Evaluation",
            "Results Analysis"
        ]
        
        self.stage_descriptions = {
            "Data Loading": "Medical data is securely loaded and validated",
            "Configuration": "Training parameters and model architecture selected",
            "Client Setup": "Medical facilities configured with local data partitions",
            "Privacy Setup": "Differential privacy mechanisms initialized",
            "Training Initiation": "Federated learning begins across all clients",
            "Fog Aggregation": "Hierarchical aggregation at fog computing nodes",
            "Global Convergence": "Model parameters converge to optimal solution",
            "Model Evaluation": "Performance metrics calculated and validated",
            "Results Analysis": "Comprehensive analysis and insights generated"
        }
        
        self.current_stage = 0
        self.stage_progress = {}
        self.journey_timeline = []
        self.interactive_elements = {}
        
    def initialize_journey(self, session_state):
        """Initialize the journey visualization based on current session state"""
        # Determine current stage based on session state
        if session_state.get('training_completed', False) or (hasattr(session_state, 'results') and session_state.results):
            self.current_stage = 8  # Results Analysis - final stage
        elif session_state.get('training_in_progress', False) or (hasattr(session_state, 'training_metrics') and session_state.training_metrics):
            # Check training progress
            if hasattr(session_state, 'training_metrics') and session_state.training_metrics:
                rounds = len(session_state.training_metrics)
                max_rounds = session_state.get('max_rounds', 20)
                
                if rounds >= max_rounds:
                    self.current_stage = 8  # Results Analysis
                elif rounds >= max(8, int(max_rounds * 0.8)):
                    self.current_stage = 7  # Model Evaluation
                elif rounds >= max(5, int(max_rounds * 0.5)):
                    self.current_stage = 6  # Global Convergence
                elif rounds >= max(2, int(max_rounds * 0.2)):
                    self.current_stage = 5  # Fog Aggregation
                else:
                    self.current_stage = 4  # Training Initiation
            else:
                self.current_stage = 4  # Training Initiation
        elif session_state.get('training_started', False):
            self.current_stage = 4  # Training Initiation
        elif session_state.get('processed_data') is not None:
            self.current_stage = 3  # Privacy Setup
        elif session_state.get('data_loaded', False):
            self.current_stage = 2  # Client Setup
        elif session_state.get('training_data') is not None:
            self.current_stage = 1  # Configuration
        else:
            self.current_stage = 0  # Data Loading
            
        # Initialize progress for all stages
        for i, stage in enumerate(self.journey_stages):
            if i < self.current_stage:
                self.stage_progress[stage] = 100
            elif i == self.current_stage:
                self.stage_progress[stage] = self._calculate_current_stage_progress(session_state)
            else:
                self.stage_progress[stage] = 0
    
    def _calculate_current_stage_progress(self, session_state) -> float:
        """Calculate progress within current stage"""
        if self.current_stage == 0:  # Data Loading
            return 100 if session_state.get('data_loaded', False) else 50
        elif self.current_stage == 1:  # Configuration
            config_items = ['num_clients', 'max_rounds', 'model_type', 'enable_dp']
            completed = sum(1 for item in config_items if session_state.get(item) is not None)
            return (completed / len(config_items)) * 100
        elif self.current_stage in [4, 5, 6]:  # Training stages
            if session_state.get('training_metrics'):
                total_rounds = session_state.get('max_rounds', 20)
                current_round = len(session_state.get('training_metrics', []))
                
                # Calculate gradual progress based on current stage
                if self.current_stage == 4:  # Training Initiation
                    # Progress from 0% to 30% based on rounds completed
                    base_progress = min(30, (current_round / max(1, total_rounds * 0.1)) * 30)
                elif self.current_stage == 5:  # Fog Aggregation
                    # Progress from 30% to 70% based on rounds completed
                    round_ratio = current_round / total_rounds
                    base_progress = 30 + min(40, round_ratio * 40)
                elif self.current_stage == 6:  # Global Convergence
                    # Progress from 70% to 95% based on rounds completed
                    round_ratio = current_round / total_rounds
                    base_progress = 70 + min(25, round_ratio * 25)
                
                return min(95, base_progress)  # Cap at 95% until truly complete
            return 10 if self.current_stage == 4 else 0
        elif self.current_stage == 8:  # Results Analysis
            # Check if training is truly completed
            if session_state.get('training_completed', False) or (hasattr(session_state, 'results') and session_state.results):
                return 100
            return 50
        else:
            return 100 if self.current_stage <= 8 else 50
    
    def create_journey_map(self):
        """Create the main interactive journey map"""
        st.subheader("🗺️ Federated Learning Journey Map")
        
        # Create journey flow diagram
        fig = self._create_journey_flow()
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive stage details
        self._create_stage_explorer()
        
    def _create_journey_flow(self):
        """Create the main journey flow visualization"""
        # Calculate positions for stages in a flowing path
        positions = self._calculate_stage_positions()
        
        fig = go.Figure()
        
        # Add connecting path
        x_path = [pos[0] for pos in positions] + [positions[0][0]]
        y_path = [pos[1] for pos in positions] + [positions[0][1]]
        
        fig.add_trace(go.Scatter(
            x=x_path,
            y=y_path,
            mode='lines',
            line=dict(color='lightblue', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add stage nodes
        for i, (stage, (x, y)) in enumerate(zip(self.journey_stages, positions)):
            progress = self.stage_progress.get(stage, 0)
            
            # Determine node color and size based on progress
            if progress == 100:
                color = '#2E8B57'  # Completed - green
                size = 25
                symbol = 'circle'
            elif progress > 0:
                color = '#FFD700'  # In progress - gold
                size = 30
                symbol = 'star'
            else:
                color = '#D3D3D3'  # Not started - gray
                size = 20
                symbol = 'circle'
            
            # Current stage highlight
            if i == self.current_stage:
                color = '#FF6B6B'  # Current - red
                size = 35
                symbol = 'diamond'
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                text=[f"{i+1}"],
                textposition="middle center",
                textfont=dict(color='white', size=12, family='Arial Black'),
                name=stage,
                hovertemplate=f"<b>{stage}</b><br>" +
                            f"Progress: {progress:.1f}%<br>" +
                            f"{self.stage_descriptions[stage]}<br>" +
                            "<extra></extra>",
                showlegend=False
            ))
            
            # Add stage labels
            fig.add_annotation(
                x=x,
                y=y-0.3,
                text=f"<b>{stage}</b><br>{progress:.0f}%",
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        
        fig.update_layout(
            title="Interactive Federated Learning Journey",
            xaxis=dict(range=[-1, 5], showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(range=[-1, 4], showgrid=False, showticklabels=False, zeroline=False),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _calculate_stage_positions(self) -> List[Tuple[float, float]]:
        """Calculate positions for stages in an engaging flow pattern"""
        positions = [
            (0, 2),    # Data Loading
            (1, 3),    # Configuration
            (2, 3.5),  # Client Setup
            (3, 3),    # Privacy Setup
            (4, 2.5),  # Training Initiation
            (3.5, 1.5), # Fog Aggregation
            (2.5, 0.5), # Global Convergence
            (1.5, 0),   # Model Evaluation
            (0.5, 1)    # Results Analysis
        ]
        return positions
    
    def _create_stage_explorer(self):
        """Create interactive stage explorer"""
        st.subheader("🔍 Stage Explorer")
        
        # Stage selector
        selected_stage = st.selectbox(
            "Select a stage to explore:",
            self.journey_stages,
            index=self.current_stage
        )
        
        # Stage details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._display_stage_details(selected_stage)
        
        with col2:
            self._display_stage_metrics(selected_stage)
    
    def _display_stage_details(self, stage: str):
        """Display detailed information for a selected stage"""
        st.markdown(f"### {stage}")
        st.write(self.stage_descriptions[stage])
        
        progress = self.stage_progress.get(stage, 0)
        
        # Progress bar
        st.progress(progress / 100)
        st.write(f"Progress: {progress:.1f}%")
        
        # Stage-specific content
        if stage == "Data Loading":
            self._data_loading_details()
        elif stage == "Configuration":
            self._configuration_details()
        elif stage == "Client Setup":
            self._client_setup_details()
        elif stage == "Privacy Setup":
            self._privacy_setup_details()
        elif stage == "Training Initiation":
            self._training_initiation_details()
        elif stage == "Fog Aggregation":
            self._fog_aggregation_details()
        elif stage == "Global Convergence":
            self._global_convergence_details()
        elif stage == "Model Evaluation":
            self._model_evaluation_details()
        elif stage == "Results Analysis":
            self._results_analysis_details()
    
    def _display_stage_metrics(self, stage: str):
        """Display metrics for the selected stage"""
        st.markdown("#### Stage Metrics")
        
        # Mock metrics based on stage
        if stage in ["Training Initiation", "Fog Aggregation", "Global Convergence"]:
            if hasattr(st.session_state, 'training_metrics') and st.session_state.training_metrics:
                latest_metrics = st.session_state.training_metrics[-1]
                st.metric("Accuracy", f"{latest_metrics.get('accuracy', 0.5):.3f}")
                st.metric("Round", latest_metrics.get('round', 1))
                st.metric("Clients", latest_metrics.get('num_clients', 5))
            else:
                st.info("Metrics will appear during training")
        else:
            # Generic progress metrics
            progress = self.stage_progress.get(stage, 0)
            st.metric("Completion", f"{progress:.1f}%")
            
            if progress == 100:
                st.success("✅ Completed")
            elif progress > 0:
                st.warning("🟡 In Progress")
            else:
                st.info("⏳ Pending")
    
    def _data_loading_details(self):
        """Details for data loading stage"""
        if st.session_state.get('data_loaded', False):
            st.success("✅ Diabetes dataset successfully loaded")
            if hasattr(st.session_state, 'training_data') and st.session_state.training_data is not None:
                data = st.session_state.training_data
                st.write(f"📊 Dataset: {data.shape[0]} patients, {data.shape[1]} features")
                st.write("🔍 Features include glucose levels, BMI, age, and other medical indicators")
        else:
            st.warning("⏳ Waiting for data to be loaded")
            st.write("📋 Next steps:")
            st.write("• Navigate to Training Control tab")
            st.write("• Dataset will be automatically loaded")
    
    def _configuration_details(self):
        """Details for configuration stage"""
        config_status = {
            "Medical Stations": st.session_state.get('num_clients'),
            "Training Rounds": st.session_state.get('max_rounds'),
            "Model Type": st.session_state.get('model_type'),
            "Differential Privacy": st.session_state.get('enable_dp'),
            "Fog Computing": st.session_state.get('enable_fog')
        }
        
        st.write("📋 Configuration Status:")
        for key, value in config_status.items():
            if value is not None:
                st.write(f"✅ {key}: {value}")
            else:
                st.write(f"⏳ {key}: Not configured")
    
    def _client_setup_details(self):
        """Details for client setup stage"""
        if st.session_state.get('processed_data'):
            num_clients = len(st.session_state.processed_data)
            st.success(f"✅ {num_clients} medical facilities configured")
            st.write("🏥 Each facility has:")
            st.write("• Private patient data partition")
            st.write("• Local model training capability")
            st.write("• Secure communication protocols")
        else:
            st.warning("⏳ Client setup pending")
    
    def _privacy_setup_details(self):
        """Details for privacy setup stage"""
        if st.session_state.get('enable_dp'):
            epsilon = st.session_state.get('epsilon', 1.0)
            delta = st.session_state.get('delta', 1e-5)
            st.success("✅ Differential privacy enabled")
            st.write(f"🔒 Privacy budget: ε={epsilon}, δ={delta:.0e}")
            st.write("🛡️ Protection mechanisms:")
            st.write("• Gaussian noise injection")
            st.write("• Gradient clipping")
            st.write("• Privacy accounting")
        else:
            st.info("ℹ️ Privacy protection disabled")
    
    def _training_initiation_details(self):
        """Details for training initiation stage"""
        if st.session_state.get('training_started'):
            st.success("✅ Federated training active")
            current_round = st.session_state.get('current_training_round', 0)
            max_rounds = st.session_state.get('max_rounds', 20)
            st.write(f"🔄 Round {current_round} of {max_rounds}")
            
            if st.session_state.get('training_metrics'):
                st.write("📊 Real-time metrics available in Live Monitoring")
        else:
            st.warning("⏳ Ready to start training")
    
    def _fog_aggregation_details(self):
        """Details for fog aggregation stage"""
        if st.session_state.get('enable_fog'):
            num_fog_nodes = st.session_state.get('num_fog_nodes', 3)
            fog_method = st.session_state.get('fog_method', 'FedAvg')
            st.success(f"✅ {num_fog_nodes} fog nodes active")
            st.write(f"⚡ Aggregation method: {fog_method}")
            st.write("🌫️ Hierarchical processing:")
            st.write("• Local client training")
            st.write("• Fog-level aggregation")
            st.write("• Global model update")
        else:
            st.info("ℹ️ Direct client-server aggregation")
    
    def _global_convergence_details(self):
        """Details for global convergence stage"""
        if st.session_state.get('training_metrics'):
            metrics = st.session_state.training_metrics
            if len(metrics) > 3:
                recent_accuracies = [m['accuracy'] for m in metrics[-3:]]
                convergence_trend = np.diff(recent_accuracies)
                
                if all(abs(trend) < 0.01 for trend in convergence_trend):
                    st.success("✅ Model converging")
                    st.write("📈 Stable performance achieved")
                else:
                    st.info("🔄 Optimization in progress")
                    st.write("📈 Performance still improving")
            else:
                st.info("🔄 Collecting convergence data")
        else:
            st.warning("⏳ Awaiting training data")
    
    def _model_evaluation_details(self):
        """Details for model evaluation stage"""
        if st.session_state.get('results'):
            results = st.session_state.results
            accuracy = results.get('accuracy', 0)
            st.success(f"✅ Model evaluation complete")
            st.write(f"🎯 Final accuracy: {accuracy:.3f}")
            st.write("📊 Comprehensive metrics:")
            st.write("• Accuracy and F1-score")
            st.write("• Confusion matrices")
            st.write("• Performance across clients")
        else:
            st.warning("⏳ Evaluation pending")
    
    def _results_analysis_details(self):
        """Details for results analysis stage"""
        if st.session_state.get('training_completed'):
            st.success("✅ Analysis complete")
            st.write("📋 Available insights:")
            st.write("• Client performance comparison")
            st.write("• Privacy-utility trade-offs")
            st.write("• Deployment recommendations")
            st.write("• Risk prediction capabilities")
        else:
            st.warning("⏳ Analysis pending training completion")
    
    def create_timeline_view(self):
        """Create timeline view of the journey"""
        st.subheader("📅 Journey Timeline")
        
        # Create timeline data
        timeline_data = []
        current_time = datetime.now()
        
        for i, stage in enumerate(self.journey_stages):
            progress = self.stage_progress.get(stage, 0)
            
            if progress == 100:
                status = "Completed"
                color = "green"
                time_offset = timedelta(minutes=i*5)
            elif progress > 0:
                status = "In Progress"
                color = "orange"
                time_offset = timedelta(minutes=i*5)
            else:
                status = "Pending"
                color = "gray"
                time_offset = timedelta(minutes=i*5 + 30)  # Future time
            
            timeline_data.append({
                'Stage': stage,
                'Status': status,
                'Progress': progress,
                'Timestamp': current_time + time_offset,
                'Description': self.stage_descriptions[stage],
                'Color': color
            })
        
        # Create timeline visualization
        fig = go.Figure()
        
        for i, data in enumerate(timeline_data):
            fig.add_trace(go.Scatter(
                x=[data['Timestamp']],
                y=[i],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=data['Color'],
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                text=[f"{data['Progress']:.0f}%"],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name=data['Stage'],
                hovertemplate=f"<b>{data['Stage']}</b><br>" +
                            f"Status: {data['Status']}<br>" +
                            f"Progress: {data['Progress']:.1f}%<br>" +
                            f"Time: {data['Timestamp'].strftime('%H:%M')}<br>" +
                            f"{data['Description']}<br>" +
                            "<extra></extra>",
                showlegend=False
            ))
        
        # Add connecting line
        y_positions = list(range(len(timeline_data)))
        timestamps = [data['Timestamp'] for data in timeline_data]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=y_positions,
            mode='lines',
            line=dict(color='lightblue', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="Federated Learning Journey Timeline",
            xaxis_title="Time",
            yaxis=dict(
                tickvals=y_positions,
                ticktext=[data['Stage'] for data in timeline_data],
                title="Journey Stages"
            ),
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_interactive_controls(self):
        """Create interactive controls for journey navigation"""
        st.subheader("🎮 Interactive Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Journey", help="Update journey status"):
                self.initialize_journey(st.session_state)
                st.rerun()
        
        with col2:
            if st.button("📊 View Current Stage", help="Jump to current active stage"):
                st.session_state.selected_journey_stage = self.current_stage
                st.rerun()
        
        with col3:
            if st.button("🎯 Focus Mode", help="Highlight next actions"):
                self._show_next_actions()
    
    def _show_next_actions(self):
        """Show recommended next actions"""
        st.subheader("🎯 Recommended Next Actions")
        
        if not st.session_state.get('data_loaded', False):
            st.info("📋 Start by loading the diabetes dataset in the Training Control tab")
        elif not st.session_state.get('training_started', False):
            st.info("🚀 Configure training parameters and start federated learning")
        elif st.session_state.get('training_in_progress', False):
            st.info("👀 Monitor training progress in the Live Monitoring tab")
        elif st.session_state.get('training_completed', False):
            st.info("📊 Explore results in Performance Analysis and Risk Assessment tabs")
        else:
            st.info("✅ All major milestones completed!")
    
    def create_progress_summary(self):
        """Create overall progress summary"""
        st.subheader("📈 Journey Progress Summary")
        
        # Calculate overall progress
        total_progress = sum(self.stage_progress.values()) / len(self.stage_progress)
        
        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completed_stages = sum(1 for p in self.stage_progress.values() if p == 100)
            st.metric("Completed Stages", f"{completed_stages}/{len(self.journey_stages)}")
        
        with col2:
            st.metric("Overall Progress", f"{total_progress:.1f}%")
        
        with col3:
            current_stage_name = self.journey_stages[self.current_stage]
            st.metric("Current Stage", current_stage_name)
        
        with col4:
            remaining_stages = len(self.journey_stages) - self.current_stage - 1
            st.metric("Remaining", remaining_stages)
        
        # Overall progress bar
        st.progress(total_progress / 100)
        
        # Journey completion status
        if total_progress == 100:
            st.success("🎉 Congratulations! You've completed the full federated learning journey!")
        elif total_progress > 75:
            st.info("🏁 You're in the final stages of the federated learning journey")
        elif total_progress > 50:
            st.info("🚀 Great progress! You're halfway through the journey")
        elif total_progress > 25:
            st.info("📈 Good start! Keep going to unlock more features")
        else:
            st.info("🌟 Welcome to federated learning! Your journey is just beginning")