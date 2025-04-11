import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(layout="wide")
def create_synthetic_gravity_data(nx=50, ny=50):
    """
    Create synthetic gravity data with a low-frequency anomaly (granite body)
    and high-frequency features plus noise
    """
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Low-frequency anomaly (granite body) - negative because granite is less dense
    granite = -5 * np.exp(-(X**2 + Y**2) / 8)
    
    # High-frequency features (shallow structures)
    shallow = (0.5 * np.sin(X*2) + 0.5 * np.cos(Y*2))
    
    # Random noise
    noise = np.random.normal(0, 0.2, (ny, nx))
    
    # Combine all components
    gravity_data = granite + shallow + noise
    return gravity_data
def classify_gravity(data):
    bins = [-np.inf, -3, -1, 1, 3, np.inf]
    labels = [
        'Porphyritic Biotite Granite',
        'Monzonitic Granite',
        'Granite',
        'Gabbro',
        'Manganese Deposit'
    ]
    idx = np.digitize(data, bins) - 1
    return idx, labels

def plot_gravity_data(original, reconstructed, residual):
    """Create a four-panel plot showing original, reconstructed, classified materials, and residual data"""
    # Get classification indices and labels
    idx, labels = classify_gravity(reconstructed)
    
    # Create custom colorscale for materials
    materials_colors = [
        '#1f77b4',  # Porphyritic Biotite Granite - blue
        '#ff7f0e',  # Monzonitic Granite - orange
        '#2ca02c',  # Granite - green
        '#d62728',  # Gabbro - red
        '#9467bd',  # Manganese Deposit - purple
    ]
    
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=(
            'Original Gravity Data', 
            'Reconstructed (Low-pass)', 
            'Geological Classification',
            'Residual (High-pass)'
        ),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
    )
    
    # Common parameters for gravity plots
    gravity_params = dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
    )
    
    # Add surfaces
    fig.add_trace(
        go.Surface(z=original, **gravity_params),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Surface(z=reconstructed, **gravity_params),
        row=1, col=2
    )
    
    # Add classified materials surface
    fig.add_trace(
        go.Surface(
            z=reconstructed,
            surfacecolor=idx,
            colorscale=[[i/4, color] for i, color in enumerate(materials_colors)],
            colorbar=dict(

                title=dict(
               text='Material Type',
               font=dict(size=14),
               side='right'
    ),
              ticktext=labels,
               tickvals=list(range(len(labels))),
               tickfont=dict(size=12),
               thickness=25,
               len=0.95,
),
            showscale=True,
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Surface(z=residual, **gravity_params),
        row=1, col=4
    )
    
    # Update layout
    fig.update_layout(
        title_text='Gravity Data SVD Analysis with Geological Classification',
        height=600,
        width=1200,
        scene=dict(zaxis_title='Gravity (mGal)'),
        scene2=dict(zaxis_title='Gravity (mGal)'),
        scene3=dict(zaxis_title='Material Type'),
        scene4=dict(zaxis_title='Gravity (mGal)')
    )
    
    return fig

def plot_singular_values(s):
    """Plot singular values and their cumulative energy"""
    cumulative_energy = np.cumsum(s) / np.sum(s) * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=np.arange(1, len(s)+1), y=s, name="Singular Values"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=np.arange(1, len(s)+1), y=cumulative_energy, 
                  name="Cumulative Energy %", line=dict(color='red')),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text='Singular Values and Cumulative Energy',
        xaxis_title='Component Number',
        yaxis_title='Singular Value',
        yaxis2_title='Cumulative Energy (%)'
    )
    
    return fig

# Streamlit app
st.title('Gravity Data SVD Analysis')
st.write("""
This app demonstrates how Singular Value Decomposition (SVD) can help separate deep geological
structures from noise in gravity data. The example shows a synthetic gravity survey over a
buried granite body (negative anomaly) with added shallow features and noise.
""")

# Add regenerate button
if st.button('Regenerate Data'):
    st.session_state.gravity_data = create_synthetic_gravity_data()
    
# Initialize data if not exists
if 'gravity_data' not in st.session_state:
    st.session_state.gravity_data = create_synthetic_gravity_data()

# Perform SVD
U, s, Vt = np.linalg.svd(st.session_state.gravity_data)

# Slider for number of components
n_components = st.slider(
    'Number of SVD components to use for reconstruction',
    min_value=1,
    max_value=len(s),
    value=5,
    help='Move the slider to see how using different numbers of singular values affects the reconstruction'
)

# Reconstruct data using selected components
s_filtered = np.zeros_like(s)
s_filtered[:n_components] = s[:n_components]
reconstructed = U @ np.diag(s_filtered) @ Vt

# Calculate residual
residual = st.session_state.gravity_data - reconstructed

# Plot gravity data
st.plotly_chart(plot_gravity_data(
    st.session_state.gravity_data,
    reconstructed,
    residual
),use_container_width=True)

# Plot singular values
st.plotly_chart(plot_singular_values(s), use_container_width=True)

# Add explanation
st.write("""
### How to interpret the results:

1. **Original Data** shows the measured gravity field with all components (deep structure + shallow features + noise).

2. **Reconstructed Data** shows the low-frequency components, primarily representing the deep granite body.
   - Using more components includes more detail but also more noise.
   - Fewer components give a smoother result focusing on the main anomaly.

3. **Residual** shows what's left after subtracting the reconstruction from the original data.
   - This includes high-frequency features and noise.
   - In real surveys, this might reveal shallow structures or highlight noise that needs filtering.

4. The **Singular Values plot** shows:
   - The relative importance of each component
   - How much of the total signal energy is captured by each additional component
""")
st.write("""
## Geological Classification

The reconstructed gravity data is classified into different geological materials based on gravity values:

- **≤ -3 mGal**: Porphyritic Biotite Granite (lowest density)
- **-3 to -1 mGal**: Monzonitic Granite
- **-1 to 1 mGal**: Granite
- **1 to 3 mGal**: Gabbro
- **> 3 mGal**: Manganese Deposit (highest density)

The classification helps interpret the geological structures based on their density contrasts.
Note that this is a simplified classification scheme for demonstration purposes.
""")