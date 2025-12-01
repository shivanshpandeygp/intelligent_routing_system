"""
Main Streamlit Application for Intelligent Routing System
Enhanced with Transfer Learning and Model Management
"""
import streamlit as st
import sys
import os
from pathlib import Path
import hashlib

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph_manager import NetworkGraph
from backend.traditional_routing import TraditionalRouting
from backend.q_learning_routing import QLearningRouting
from backend.dqn_routing import DQNRouting
from backend.performance_metrics import PerformanceMetrics
from backend.config import Config
from frontend.visualizer import NetworkVisualizer
from frontend.dashboard import PerformanceDashboard
import numpy as np
import logging
import pickle
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Intelligent RL-Based Routing",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'network' not in st.session_state:
    st.session_state.network = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    """Main application"""

    st.markdown('<p class="main-header">🌐 Intelligent Reinforcement Learning Based Routing</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem;'>
        Dynamic packet routing optimization using Q-Learning and Deep Q-Learning with Transfer Learning,
        compared with traditional algorithms (Dijkstra & Bellman-Ford)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/network.png", width=100)
        st.title("⚙️ Configuration")

        menu = st.radio(
            "Navigation",
            ["🏠 Home", "📊 Network Setup", "🤖 Train Models",
             "🔍 Route Discovery", "📈 Performance Dashboard",
             "💾 Export Results", "🧠 Model Manager"]
        )

        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system demonstrates RL-based intelligent routing with **Transfer Learning** 
        that adapts to dynamic network conditions including link failures and congestion.
        
        **New**: Models learn continuously and improve over time!
        """)

    # Route to appropriate page
    if menu == "🏠 Home":
        home_page()
    elif menu == "📊 Network Setup":
        network_setup_page()
    elif menu == "🤖 Train Models":
        train_models_page()
    elif menu == "🔍 Route Discovery":
        route_discovery_page()
    elif menu == "📈 Performance Dashboard":
        performance_dashboard_page()
    elif menu == "💾 Export Results":
        export_results_page()
    elif menu == "🧠 Model Manager":
        model_manager_page()

def home_page():
    """Home page"""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🎯 Project Overview")
        st.markdown("""
        ### Key Features
        
        ✅ **Adaptive Learning**: RL agents learn optimal routing policies through exploration
        
        ✅ **Transfer Learning**: Pre-trained models work on any network immediately
        
        ✅ **Continual Learning**: Models get smarter with each training session
        
        ✅ **Dynamic Adaptation**: Handles link failures and congestion in real-time
        
        ✅ **Multi-Algorithm**: Compares Q-Learning, DQN, Dijkstra, and Bellman-Ford
        
        ✅ **Comprehensive Metrics**: Path cost, latency, energy efficiency, PDR
        
        ✅ **Interactive Visualization**: Real-time hop-by-hop packet routing animations
        
        ✅ **Scalable Architecture**: Supports mesh, ring, tree, and random topologies
        """)

        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Network Setup**: Create or upload a network topology
        2. **Check Models**: View pre-trained model status in Model Manager
        3. **Train Models**: Train Q-Learning and DQN (optional if pre-trained exists)
        4. **Route Discovery**: Find paths using different algorithms
        5. **Compare Performance**: Analyze metrics and visualizations
        6. **Export Results**: Save results as PDF/CSV
        """)

        st.markdown("### 🧠 Transfer Learning Feature")
        st.markdown("""
        **How It Works:**
        
        1. **First Time**: Train models on any network → Creates global pre-trained models
        2. **New Network**: Models automatically load pre-trained knowledge
        3. **Zero-Shot Routing**: Use pre-trained models immediately without retraining
        4. **Incremental Learning**: Each training session improves the global model
        
        **Benefits:**
        - 🚀 Faster convergence on new networks
        - 🎯 Better initial performance without training
        - 📈 Models get smarter over time across all networks
        - ⏰ Save training time on similar topologies
        - 🔄 Continuous improvement with each session
        """)

    with col2:
        st.markdown("## 📊 System Stats")

        if st.session_state.network:
            net = st.session_state.network
            st.metric("Network Nodes", net.num_nodes)
            st.metric("Network Edges", net.graph.number_of_edges())
            st.metric("Topology Type", net.topology.upper())

            if st.session_state.trained_models:
                st.success(f"✓ {len(st.session_state.trained_models)} models loaded")
            else:
                st.warning("⚠ No models loaded for this network")
        else:
            st.warning("⚠ No network created yet")

        st.markdown("---")

        # Global model status
        st.markdown("### 🌍 Global Models")

        ql_global = Path("models/pretrained/q_learning_global.pkl")
        dqn_global = Path("models/pretrained/dqn_global.pt")

        if ql_global.exists():
            st.success("✅ Q-Learning Global")
        else:
            st.info("ℹ️ Q-Learning: Not trained")

        if dqn_global.exists():
            st.success("✅ DQN Global")
        else:
            st.info("ℹ️ DQN: Not trained")

        st.markdown("---")
        st.markdown("### 🔧 Technologies")
        st.markdown("""
        - **Backend**: Python, PyTorch
        - **RL Algorithms**: Q-Learning, DQN
        - **Transfer Learning**: Global + Network-specific
        - **Frontend**: Streamlit
        - **Visualization**: NetworkX, Plotly
        """)


def network_setup_page():
    """Network setup page with custom network builder"""

    st.markdown("## 📊 Network Topology Setup")

    tab1, tab2, tab3 = st.tabs(["Create Network", "Custom Network Builder", "Load Network"])

    with tab1:
        st.markdown("### Create New Network")

        col1, col2 = st.columns(2)

        with col1:
            topology = st.selectbox(
                "Select Topology",
                ["mesh", "ring", "tree", "random"],
                help="Choose network topology type"
            )

            num_nodes = st.slider(
                "Number of Nodes",
                min_value=5,
                max_value=50,
                value=10,
                help="Total number of nodes in the network"
            )

        with col2:
            sparsity = st.slider(
                "Connection Density",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Percentage of possible connections (lower = sparser)"
            )

            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                help="For reproducible network generation"
            )

        if st.button("🔨 Create Network", type="primary", key="create_network_btn"):
            with st.spinner("Creating network topology..."):
                try:
                    np.random.seed(seed)
                    network = NetworkGraph(
                        num_nodes=num_nodes,
                        topology=topology,
                        sparsity=sparsity
                    )
                    st.session_state.network = network
                    st.session_state.trained_models = {}
                    st.success(f"✅ Network created successfully! "
                               f"{network.graph.number_of_edges()} edges generated.")

                    # Check for network-specific models
                    import hashlib
                    network_id = hashlib.md5(
                        f"{topology}_{num_nodes}_{network.graph.number_of_edges()}".encode()
                    ).hexdigest()[:8]

                    ql_specific = Path(f"models/network_specific/network_{network_id}_ql.pkl")
                    dqn_specific = Path(f"models/network_specific/network_{network_id}_dqn.pt")

                    if ql_specific.exists() or dqn_specific.exists():
                        st.info("📌 Network-specific pre-trained models found!")

                    # Visualize
                    visualizer = NetworkVisualizer(network)
                    fig = visualizer.plot_network()
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error creating network: {str(e)}")
                    logger.error(f"Network creation error: {e}", exc_info=True)

    with tab2:
        st.markdown("### 🔧 Custom Network Builder")
        st.info("💡 **Build your network from scratch!** Define nodes and edges with custom weights.")

        # Initialize session state for custom network builder
        if 'custom_num_nodes' not in st.session_state:
            st.session_state.custom_num_nodes = 5
        if 'custom_edges' not in st.session_state:
            st.session_state.custom_edges = []
        if 'current_edge_step' not in st.session_state:
            st.session_state.current_edge_step = 0

        # Step 1: Number of nodes
        st.markdown("#### Step 1: Define Number of Nodes")
        num_custom_nodes = st.number_input(
            "Number of Nodes",
            min_value=2,
            max_value=30,
            value=st.session_state.custom_num_nodes,
            help="How many nodes in your network?"
        )

        if num_custom_nodes != st.session_state.custom_num_nodes:
            st.session_state.custom_num_nodes = num_custom_nodes
            st.session_state.custom_edges = []
            st.session_state.current_edge_step = 0

        st.markdown("---")

        # Step 2: Define edges
        st.markdown("#### Step 2: Define Edges")
        st.write(f"Define connections between {num_custom_nodes} nodes (Node IDs: 0 to {num_custom_nodes - 1})")

        # Display current edges
        if st.session_state.custom_edges:
            st.markdown("**📋 Current Edges:**")
            edges_df = pd.DataFrame(st.session_state.custom_edges,
                                    columns=['Source', 'Destination', 'Weight'])
            st.dataframe(edges_df, use_container_width=True, hide_index=True)

            st.info(f"✓ {len(st.session_state.custom_edges)} edges defined")
        else:
            st.info("ℹ️ No edges defined yet. Add edges below.")

        st.markdown("---")

        # Add new edge
        st.markdown("**➕ Add New Edge**")

        add_reverse = st.checkbox(
            "Add reverse edge automatically (bidirectional)",
            value=True,
            help="Automatically add edge in opposite direction with same weight",
            key="bidirectional_checkbox"
        )

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            edge_source = st.number_input(
                "Source Node",
                min_value=0,
                max_value=num_custom_nodes - 1,
                value=0,
                key="edge_source"
            )

        with col2:
            edge_dest = st.number_input(
                "Destination Node",
                min_value=0,
                max_value=num_custom_nodes - 1,
                value=1,
                key="edge_dest"
            )

        with col3:
            edge_weight = st.number_input(
                "Edge Weight",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                key="edge_weight"
            )

        with col4:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("➕ Add", key="add_edge_btn"):
                if edge_source == edge_dest:
                    st.error("❌ Source and destination cannot be the same!")
                else:
                    edges_added = []

                    # Check if forward edge exists
                    forward_exists = any(
                        e[0] == edge_source and e[1] == edge_dest
                        for e in st.session_state.custom_edges
                    )

                    # Add forward edge if not exists
                    if not forward_exists:
                        st.session_state.custom_edges.append((edge_source, edge_dest, edge_weight))
                        edges_added.append(f"{edge_source} → {edge_dest}")

                    # Add reverse edge if bidirectional is checked
                    if add_reverse:
                        reverse_exists = any(
                            e[0] == edge_dest and e[1] == edge_source
                            for e in st.session_state.custom_edges
                        )
                        if not reverse_exists:
                            st.session_state.custom_edges.append((edge_dest, edge_source, edge_weight))
                            edges_added.append(f"{edge_dest} → {edge_source}")

                    # Show result
                    if edges_added:
                        st.success(f"✅ Added edge(s): {', '.join(edges_added)} (weight: {edge_weight})")
                        st.rerun()
                    else:
                        st.warning("⚠️ Edge(s) already exist!")

        # Show info about current mode
        if add_reverse:
            st.caption("🔄 Bidirectional mode ON: Both A→B and B→A will be added")
        else:
            st.caption("➡️ Unidirectional mode: Only A→B will be added")

        st.markdown("---")

        # Quick actions
        st.markdown("**⚡ Quick Actions**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔗 Complete Graph", help="Connect all nodes to all other nodes"):
                st.session_state.custom_edges = []
                for i in range(num_custom_nodes):
                    for j in range(num_custom_nodes):
                        if i != j:
                            weight = np.random.randint(5, 20)
                            st.session_state.custom_edges.append((i, j, float(weight)))
                st.success(f"✅ Created complete graph with {len(st.session_state.custom_edges)} edges!")
                st.rerun()

        with col2:
            if st.button("🔄 Ring Graph", help="Connect nodes in a ring"):
                st.session_state.custom_edges = []
                for i in range(num_custom_nodes):
                    next_node = (i + 1) % num_custom_nodes
                    weight = np.random.randint(5, 20)
                    st.session_state.custom_edges.append((i, next_node, float(weight)))
                    st.session_state.custom_edges.append((next_node, i, float(weight)))
                st.success(f"✅ Created ring graph!")
                st.rerun()

        with col3:
            if st.button("🌟 Star Graph", help="Connect all nodes to node 0"):
                st.session_state.custom_edges = []
                for i in range(1, num_custom_nodes):
                    weight = np.random.randint(5, 20)
                    st.session_state.custom_edges.append((0, i, float(weight)))
                    st.session_state.custom_edges.append((i, 0, float(weight)))
                st.success(f"✅ Created star graph!")
                st.rerun()

        with col4:
            if st.button("🗑️ Clear All", help="Remove all edges"):
                st.session_state.custom_edges = []
                st.success("✅ All edges cleared!")
                st.rerun()

        st.markdown("---")

        # Remove specific edge
        if st.session_state.custom_edges:
            st.markdown("**🗑️ Remove Edge**")
            edge_to_remove = st.selectbox(
                "Select edge to remove",
                options=range(len(st.session_state.custom_edges)),
                format_func=lambda
                    i: f"{st.session_state.custom_edges[i][0]} → {st.session_state.custom_edges[i][1]} (weight: {st.session_state.custom_edges[i][2]})",
                key="edge_to_remove"
            )

            if st.button("🗑️ Remove Selected Edge", key="remove_edge_btn"):
                removed = st.session_state.custom_edges.pop(edge_to_remove)
                st.success(f"✅ Removed edge: {removed[0]} → {removed[1]}")
                st.rerun()

        st.markdown("---")

        # Create custom network
        st.markdown("#### Step 3: Create Network")

        if len(st.session_state.custom_edges) == 0:
            st.warning("⚠️ Please add at least one edge before creating the network.")
        else:
            st.success(
                f"✅ Ready to create network with {num_custom_nodes} nodes and {len(st.session_state.custom_edges)} edges")

            if st.button("🚀 Create Custom Network", type="primary", key="create_custom_btn"):
                with st.spinner("Creating custom network..."):
                    try:
                        # Create custom network
                        network = NetworkGraph.create_custom(
                            num_nodes=num_custom_nodes,
                            edges=st.session_state.custom_edges
                        )

                        st.session_state.network = network
                        st.session_state.trained_models = {}

                        st.success(f"✅ Custom network created successfully!")
                        st.info(f"📊 Network Stats:\n\n"
                                f"• Nodes: {network.num_nodes}\n\n"
                                f"• Edges: {network.graph.number_of_edges()}\n\n"
                                f"• Topology: Custom\n\n"
                                f"• Connected: {'Yes ✓' if network.is_connected() else 'No ✗'}")

                        # Visualize
                        st.markdown("### 🗺️ Network Visualization")
                        visualizer = NetworkVisualizer(network)
                        fig = visualizer.plot_network(title="Custom Network Topology")
                        st.plotly_chart(fig, use_container_width=True)

                        # Network statistics
                        st.markdown("### 📈 Network Statistics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Nodes", network.num_nodes)
                        with col2:
                            st.metric("Edges", network.graph.number_of_edges())
                        with col3:
                            avg_degree = network.graph.number_of_edges() / network.num_nodes if network.num_nodes > 0 else 0
                            st.metric("Avg Degree", f"{avg_degree:.2f}")
                        with col4:
                            is_connected = "Yes" if network.is_connected() else "No"
                            st.metric("Connected", is_connected)

                        # Edge list
                        with st.expander("📋 View All Edges"):
                            edge_data = []
                            for u, v, data in network.graph.edges(data=True):
                                edge_data.append({
                                    'Source': u,
                                    'Destination': v,
                                    'Weight': data['weight'],
                                    'Latency': network.edge_metrics[(u, v)]['latency'],
                                    'Energy': network.edge_metrics[(u, v)]['energy']
                                })
                            edge_df = pd.DataFrame(edge_data)
                            st.dataframe(edge_df, use_container_width=True, hide_index=True)

                    except Exception as e:
                        st.error(f"❌ Error creating custom network: {str(e)}")
                        logger.error(f"Custom network creation error: {e}", exc_info=True)

    with tab3:
        st.markdown("### Load Existing Network")

        uploaded_file = st.file_uploader(
            "Upload Network File (.pkl)",
            type=['pkl'],
            help="Upload a previously saved network"
        )

        if uploaded_file:
            if st.button("📂 Load Network", key="load_network_btn"):
                try:
                    import pickle
                    network = pickle.load(uploaded_file)
                    st.session_state.network = network
                    st.success("✅ Network loaded successfully!")

                    visualizer = NetworkVisualizer(network)
                    fig = visualizer.plot_network()
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error loading network: {str(e)}")

    # Display current network statistics (if network exists)
    if st.session_state.network:
        st.markdown("---")
        st.markdown("### 📊 Current Network Statistics")

        net = st.session_state.network
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nodes", net.num_nodes)
        with col2:
            st.metric("Edges", net.graph.number_of_edges())
        with col3:
            avg_degree = 2 * net.graph.number_of_edges() / net.num_nodes if net.num_nodes > 0 else 0
            st.metric("Avg Degree", f"{avg_degree:.2f}")
        with col4:
            is_connected = "Yes ✓" if net.is_connected() else "No ✗"
            st.metric("Connected", is_connected)

        # Save network option
        st.markdown("---")
        if st.button("💾 Save Current Network", key="save_network_btn"):
            try:
                import tempfile
                import pickle as pkl
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as f:
                    pkl.dump(net, f)
                    temp_file_path = f.name

                with open(temp_file_path, 'rb') as file:
                    network_data = file.read()

                import os
                os.unlink(temp_file_path)

                st.download_button(
                    label="📥 Download Network File",
                    data=network_data,
                    file_name=f"network_{net.topology}_{net.num_nodes}nodes.pkl",
                    mime="application/octet-stream",
                    key = "download_network_file"
                )

                st.success("✅ Network file ready for download!")

            except Exception as e:
                st.error(f"Error saving network: {e}")
                logger.error(f"Network save error: {e}", exc_info=True)


def train_models_page():
    """Model training page with transfer learning"""

    if not st.session_state.network:
        st.warning("⚠️ Please create a network first!")
        return

    st.markdown("## 🤖 Train Reinforcement Learning Models")

    # Show global model status
    st.markdown("### 📊 Global Pre-trained Models Status")

    col1, col2 = st.columns(2)

    with col1:
        ql_pretrained_path = Path("models/pretrained/q_learning_global.pkl")
        if ql_pretrained_path.exists():
            try:
                with open(ql_pretrained_path, 'rb') as f:
                    ql_data = pickle.load(f)
                st.success("✅ Q-Learning Global Model Available")
                st.info(f"🎯 Version: {ql_data.get('version', 'N/A')}\n\n"
                       f"📚 Total Episodes: {ql_data.get('total_episodes', 0):,}\n\n"
                       f"🕒 Last Updated: {ql_data.get('last_updated', 'N/A')}\n\n"
                       f"💾 State-Action Pairs: {sum(len(actions) for actions in ql_data.get('q_table', {}).values()):,}")
            except:
                st.warning("⚠️ Q-Learning model exists but couldn't load stats")
        else:
            st.info("ℹ️ No Q-Learning global model yet.\n\nTrain to create one!")

    with col2:
        dqn_pretrained_path = Path("models/pretrained/dqn_global.pt")
        if dqn_pretrained_path.exists():
            try:
                dqn_data = torch.load(dqn_pretrained_path, map_location='cpu')
                st.success("✅ DQN Global Model Available")
                st.info(f"🎯 Version: {dqn_data.get('version', 'N/A')}\n\n"
                       f"📚 Total Episodes: {dqn_data.get('total_episodes', 0):,}\n\n"
                       f"🕒 Last Updated: {dqn_data.get('last_updated', 'N/A')}\n\n"
                       f"🧠 Neural Network: Trained")
            except:
                st.warning("⚠️ DQN model exists but couldn't load stats")
        else:
            st.info("ℹ️ No DQN global model yet.\n\nTrain to create one!")

    st.markdown("---")

    network = st.session_state.network

    col1, col2 = st.columns(2)

    with col1:
        source = st.selectbox(
            "Source Node",
            range(network.num_nodes),
            help="Starting node for training"
        )

    with col2:
        destination = st.selectbox(
            "Destination Node",
            range(network.num_nodes),
            index=min(network.num_nodes-1, network.num_nodes),
            help="Target node for training"
        )

    if source == destination:
        st.error("❌ Source and destination must be different!")
        return

    st.markdown("---")

    tab1, tab2 = st.tabs(["Q-Learning", "Deep Q-Network (DQN)"])

    with tab1:
        st.markdown("### Q-Learning Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            ql_episodes = st.number_input(
                "Training Episodes",
                min_value=100,
                max_value=20000,
                value=1000,
                step=100
            )

        with col2:
            ql_lr = st.number_input(
                "Learning Rate",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
            )

        with col3:
            ql_gamma = st.number_input(
                "Discount Factor",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01
            )

        save_specific = st.checkbox(
            "💾 Save network-specific model",
            value=True,
            help="Save a dedicated model for this exact network topology"
        )

        if st.button("🎯 Train Q-Learning", type="primary"):
            with st.spinner(f"Training Q-Learning for {ql_episodes} episodes..."):
                try:
                    config = Config()
                    config.QL_EPISODES = ql_episodes
                    config.QL_LEARNING_RATE = ql_lr
                    config.QL_DISCOUNT_FACTOR = ql_gamma

                    ql_agent = QLearningRouting(network, config, load_pretrained=True)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Training
                    ql_agent.train(source, destination, ql_episodes, save_specific=save_specific)

                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")

                    st.session_state.trained_models['q_learning'] = ql_agent

                    stats = ql_agent.get_q_table_stats()

                    st.success(f"✅ Q-Learning trained successfully!")
                    st.info(f"📊 **Training Summary**\n\n"
                           f"• Episodes: {ql_episodes:,}\n\n"
                           f"• Total Episodes (cumulative): {stats['total_training_episodes']:,}\n\n"
                           f"• Q-Table Size: {stats['total_state_action_pairs']:,} entries\n\n"
                           f"• Network-Specific Saved: {'Yes ✓' if save_specific else 'No'}\n\n"
                           f"• Global Model Updated: Yes ✓")

                    # Plot training progress
                    st.markdown("### 📊 Training Progress")
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ql_agent.training_history['episodes'],
                        y=ql_agent.training_history['rewards'],
                        mode='lines',
                        name='Reward',
                        line=dict(color='blue')
                    ))
                    fig.update_layout(
                        title="Q-Learning Training Rewards",
                        xaxis_title="Episode",
                        yaxis_title="Total Reward",
                        hovermode='x'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    logger.error(f"Q-Learning training error: {e}", exc_info=True)

    with tab2:
        st.markdown("### Deep Q-Network Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            dqn_episodes = st.number_input(
                "Training Episodes",
                min_value=100,
                max_value=10000,
                value=500,
                step=50,
                key="dqn_episodes"
            )

        with col2:
            dqn_lr = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key="dqn_lr"
            )

        with col3:
            dqn_hidden = st.number_input(
                "Hidden Layer Size",
                min_value=64,
                max_value=512,
                value=128,
                step=64
            )

        save_specific_dqn = st.checkbox(
            "💾 Save network-specific model",
            value=True,
            help="Save a dedicated model for this exact network topology",
            key="save_specific_dqn"
        )

        if st.button("🎯 Train DQN", type="primary", key="train_dqn_btn"):
            with st.spinner(f"Training DQN for {dqn_episodes} episodes..."):
                try:
                    config = Config()
                    config.DQN_EPISODES = dqn_episodes
                    config.DQN_LEARNING_RATE = dqn_lr
                    config.DQN_HIDDEN_SIZE = dqn_hidden

                    dqn_agent = DQNRouting(network, config, load_pretrained=True)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    dqn_agent.train(source, destination, dqn_episodes, save_specific=save_specific_dqn)

                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")

                    st.session_state.trained_models['dqn'] = dqn_agent

                    st.success(f"✅ DQN trained successfully!")
                    st.info(f"📊 **Training Summary**\n\n"
                           f"• Episodes: {dqn_episodes:,}\n\n"
                           f"• Total Episodes (cumulative): {dqn_agent.total_training_episodes:,}\n\n"
                           f"• Hidden Layer Size: {dqn_hidden}\n\n"
                           f"• Network-Specific Saved: {'Yes ✓' if save_specific_dqn else 'No'}\n\n"
                           f"• Global Model Updated: Yes ✓")

                    # Plot training progress
                    st.markdown("### 📊 Training Progress")
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Rewards', 'Loss')
                    )

                    fig.add_trace(go.Scatter(
                        x=dqn_agent.training_history['episodes'],
                        y=dqn_agent.training_history['rewards'],
                        mode='lines',
                        name='Reward',
                        line=dict(color='blue')
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=dqn_agent.training_history['episodes'],
                        y=dqn_agent.training_history['losses'],
                        mode='lines',
                        name='Loss',
                        line=dict(color='red')
                    ), row=1, col=2)

                    fig.update_xaxes(title_text="Episode", row=1, col=1)
                    fig.update_xaxes(title_text="Episode", row=1, col=2)
                    fig.update_yaxes(title_text="Total Reward", row=1, col=1)
                    fig.update_yaxes(title_text="Loss", row=1, col=2)

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    logger.error(f"DQN training error: {e}", exc_info=True)

def route_discovery_page():
    """Route discovery and visualization page"""

    if not st.session_state.network:
        st.warning("⚠️ Please create a network first!")
        return

    st.markdown("## 🔍 Route Discovery & Visualization")

    network = st.session_state.network

    col1, col2, col3 = st.columns(3)

    with col1:
        source = st.selectbox(
            "Source Node",
            range(network.num_nodes),
            key="route_source"
        )

    with col2:
        destination = st.selectbox(
            "Destination Node",
            range(network.num_nodes),
            index=min(network.num_nodes-1, network.num_nodes),
            key="route_dest"
        )

    with col3:
        algorithms = st.multiselect(
            "Select Algorithms",
            ["Dijkstra", "Bellman-Ford", "Q-Learning", "DQN"],
            default=["Dijkstra"],
            help="Choose algorithms to compare"
        )

    if source == destination:
        st.error("❌ Source and destination must be different!")
        return

    if st.button("🚀 Find Routes", type="primary"):
        results = {}

        with st.spinner("Computing routes..."):
            # Traditional algorithms
            traditional = TraditionalRouting(network)

            if "Dijkstra" in algorithms:
                path, cost, time_taken = traditional.dijkstra(source, destination)
                results['Dijkstra'] = {'path': path, 'cost': cost, 'time': time_taken}

            if "Bellman-Ford" in algorithms:
                path, cost, time_taken = traditional.bellman_ford(source, destination)
                results['Bellman-Ford'] = {'path': path, 'cost': cost, 'time': time_taken}

            # RL algorithms
            if "Q-Learning" in algorithms:
                if 'q_learning' in st.session_state.trained_models:
                    ql_agent = st.session_state.trained_models['q_learning']
                    path, cost, time_taken = ql_agent.find_path(source, destination)
                    results['Q-Learning'] = {'path': path, 'cost': cost, 'time': time_taken}
                else:
                    # Try loading pre-trained model
                    try:
                        config = Config()
                        ql_agent = QLearningRouting(network, config, load_pretrained=True)
                        path, cost, time_taken = ql_agent.find_path(source, destination)
                        results['Q-Learning'] = {'path': path, 'cost': cost, 'time': time_taken}
                        st.info("ℹ️ Using pre-trained Q-Learning model (not trained on this session)")
                    except:
                        st.warning("⚠️ Q-Learning model not available. Please train first!")

            if "DQN" in algorithms:
                if 'dqn' in st.session_state.trained_models:
                    dqn_agent = st.session_state.trained_models['dqn']
                    path, cost, time_taken = dqn_agent.find_path(source, destination)
                    results['DQN'] = {'path': path, 'cost': cost, 'time': time_taken}
                else:
                    # Try loading pre-trained model
                    try:
                        config = Config()
                        dqn_agent = DQNRouting(network, config, load_pretrained=True)
                        path, cost, time_taken = dqn_agent.find_path(source, destination)
                        results['DQN'] = {'path': path, 'cost': cost, 'time': time_taken}
                        st.info("ℹ️ Using pre-trained DQN model (not trained on this session)")
                    except:
                        st.warning("⚠️ DQN model not available. Please train first!")

        st.session_state.results = results

        if results:
            st.success(f"✅ Found {len(results)} routes!")

            # Display results
            st.markdown("### 📊 Route Comparison")

            cols = st.columns(len(results))
            for idx, (algo_name, algo_results) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"**{algo_name}**")
                    st.metric("Path Cost", f"{algo_results['cost']:.2f}")

                    # Enhanced time display
                    time_ms = algo_results['time'] * 1000
                    if time_ms < 0.01:
                        st.metric("Time", f"{algo_results['time']*1000000:.2f} μs")
                    else:
                        st.metric("Time", f"{time_ms:.4f} ms")

                    st.metric("Hops", len(algo_results['path']) - 1 if algo_results['path'] else 0)

            # Visualize paths
            st.markdown("### 🗺️ Path Visualization")

            visualizer = NetworkVisualizer(network)

            for algo_name, algo_results in results.items():
                with st.expander(f"📍 {algo_name} Path"):
                    if algo_results['path']:
                        st.write(f"**Path:** {' → '.join(map(str, algo_results['path']))}")
                        fig = visualizer.plot_path(algo_results['path'],
                                                   title=f"{algo_name} Route")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ No path found!")

def performance_dashboard_page():
    """Performance comparison dashboard"""

    if not st.session_state.results:
        st.warning("⚠️ Please discover routes first!")
        return

    st.markdown("## 📈 Performance Comparison Dashboard")

    results = st.session_state.results
    network = st.session_state.network

    metrics_calc = PerformanceMetrics(network)
    comparison = metrics_calc.compare_algorithms(results)

    # Display dashboard
    dashboard = PerformanceDashboard(comparison, network)
    dashboard.render()

def export_results_page():
    """Export results page"""

    if not st.session_state.results:
        st.warning("⚠️ Please discover routes first!")
        return

    st.markdown("## 💾 Export Results")

    results = st.session_state.results
    network = st.session_state.network

    metrics_calc = PerformanceMetrics(network)
    comparison = metrics_calc.compare_algorithms(results)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 CSV Export")
        st.write("Export metrics as CSV file")

        if st.button("📥 Download CSV", type="primary"):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
                metrics_calc.export_to_csv(comparison, f.name)

                with open(f.name, 'r') as file:
                    csv_data = file.read()

                st.download_button(
                    label="💾 Save CSV File",
                    data=csv_data,
                    file_name="routing_metrics.csv",
                    mime="text/csv"
                )

    with col2:
        st.markdown("### 📑 PDF Export")
        st.write("Export detailed report as PDF")

        if st.button("📥 Generate PDF", type="primary"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                    metrics_calc.export_to_pdf(comparison, f.name)

                    with open(f.name, 'rb') as file:
                        pdf_data = file.read()

                    st.download_button(
                        label="💾 Save PDF Report",
                        data=pdf_data,
                        file_name="routing_report.pdf",
                        mime="application/pdf"
                    )
            except ImportError:
                st.error("❌ reportlab not installed. Please install: pip install reportlab")

    st.markdown("---")
    st.markdown("### 📊 Performance Report Preview")

    report = metrics_calc.generate_performance_report(comparison)
    st.code(report, language="text")

def model_manager_page():
    """Model management page"""

    st.markdown("## 🧠 Model Manager")
    st.markdown("View and manage pre-trained models")

    tab1, tab2 = st.tabs(["Global Models", "Network-Specific Models"])

    with tab1:
        st.markdown("### 🌍 Global Pre-trained Models")
        st.info("These models work across all network topologies and improve with each training session.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Q-Learning Global Model")
            ql_path = Path("models/pretrained/q_learning_global.pkl")

            if ql_path.exists():
                try:
                    with open(ql_path, 'rb') as f:
                        ql_data = pickle.load(f)

                    st.success("✅ Model Exists")
                    st.metric("Version", ql_data.get('version', 'N/A'))
                    st.metric("Total Episodes", f"{ql_data.get('total_episodes', 0):,}")
                    st.metric("State-Action Pairs", f"{sum(len(actions) for actions in ql_data.get('q_table', {}).values()):,}")
                    st.text(f"Last Updated: {ql_data.get('last_updated', 'N/A')}")

                    if st.button("🗑️ Delete Q-Learning Global", key="del_ql_global"):
                        ql_path.unlink()
                        st.success("✅ Deleted successfully! Refresh page.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading model: {e}")
            else:
                st.warning("⚠️ No global model found")
                st.info("Train any network to create global model")

        with col2:
            st.markdown("#### DQN Global Model")
            dqn_path = Path("models/pretrained/dqn_global.pt")

            if dqn_path.exists():
                try:
                    dqn_data = torch.load(dqn_path, map_location='cpu')

                    st.success("✅ Model Exists")
                    st.metric("Version", dqn_data.get('version', 'N/A'))
                    st.metric("Total Episodes", f"{dqn_data.get('total_episodes', 0):,}")
                    st.text(f"Last Updated: {dqn_data.get('last_updated', 'N/A')}")

                    if st.button("🗑️ Delete DQN Global", key="del_dqn_global"):
                        dqn_path.unlink()
                        st.success("✅ Deleted successfully! Refresh page.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading model: {e}")
            else:
                st.warning("⚠️ No global model found")
                st.info("Train any network to create global model")

    with tab2:
        st.markdown("### 📁 Network-Specific Models")
        st.info("These models are optimized for specific network topologies.")

        specific_dir = Path("models/network_specific")

        if specific_dir.exists():
            ql_models = list(specific_dir.glob("*_ql.pkl"))
            dqn_models = list(specific_dir.glob("*_dqn.pt"))

            if ql_models or dqn_models:
                st.markdown(f"**Found {len(ql_models)} Q-Learning and {len(dqn_models)} DQN models**")

                # Display Q-Learning models
                if ql_models:
                    st.markdown("#### Q-Learning Models")
                    for model_path in ql_models:
                        with st.expander(f"📦 {model_path.name}"):
                            try:
                                with open(model_path, 'rb') as f:
                                    model_data = pickle.load(f)

                                net_info = model_data.get('network_info', {})
                                st.write(f"**Topology**: {net_info.get('topology', 'N/A')}")
                                st.write(f"**Nodes**: {net_info.get('num_nodes', 'N/A')}")
                                st.write(f"**Edges**: {net_info.get('num_edges', 'N/A')}")
                                st.write(f"**Episodes Trained**: {model_data.get('total_episodes', 0):,}")
                                st.write(f"**Last Updated**: {model_data.get('last_updated', 'N/A')}")

                                if st.button(f"🗑️ Delete", key=f"del_{model_path.name}"):
                                    model_path.unlink()
                                    st.success("✅ Deleted!")
                                    st.rerun()

                            except Exception as e:
                                st.error(f"Error: {e}")

                # Display DQN models
                if dqn_models:
                    st.markdown("#### DQN Models")
                    for model_path in dqn_models:
                        with st.expander(f"📦 {model_path.name}"):
                            try:
                                model_data = torch.load(model_path, map_location='cpu')

                                net_info = model_data.get('network_info', {})
                                st.write(f"**Topology**: {net_info.get('topology', 'N/A')}")
                                st.write(f"**Nodes**: {net_info.get('num_nodes', 'N/A')}")
                                st.write(f"**Edges**: {net_info.get('num_edges', 'N/A')}")
                                st.write(f"**Episodes Trained**: {model_data.get('total_episodes', 0):,}")
                                st.write(f"**Last Updated**: {model_data.get('last_updated', 'N/A')}")

                                if st.button(f"🗑️ Delete", key=f"del_{model_path.name}"):
                                    model_path.unlink()
                                    st.success("✅ Deleted!")
                                    st.rerun()

                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.info("📭 No network-specific models found yet.\n\nTrain with 'Save network-specific model' enabled.")
        else:
            st.info("📭 No network-specific models directory found yet.")

if __name__ == "__main__":
    import hashlib  # Add this import at the top with other imports
    main()
