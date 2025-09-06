import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.special import hermite
from math import factorial  # Using math.factorial which is standard

# Page config
st.set_page_config(page_title="Quantum Mechanics Visualizer", page_icon="⚛️", layout="wide")

# Title
st.title("⚛️ Quantum Mechanics Visualizer")
st.markdown("Interactive visualizations of quantum mechanical systems")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📦 Particle in a Box", "⚡ Quantum Tunneling", "🌊 Harmonic Oscillator"])

# ============================================
# TAB 1: PARTICLE IN A BOX
# ============================================
with tab1:
    st.header("Quantum Particle in a Box")
    st.write("Visualizing wavefunctions ψₙ(x) and probability densities |ψₙ(x)|²")
    
    # Create columns for controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        L = st.slider("Box Length (L)", min_value=1, max_value=10, value=5, step=1)
    with col2:
        n = st.slider("Quantum Number (n)", min_value=1, max_value=10, value=4, step=1)
    with col3:
        show_box = st.checkbox("Show box walls", value=True)
        show_equations = st.checkbox("Show equations", value=True)
    
    # Display equations
    if show_equations:
        st.markdown("### 📐 Mathematical Description")
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r"\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)")
            st.caption("Wavefunction")
        
        with col2:
            st.latex(r"E_n = \frac{n^2 \pi^2 \hbar^2}{2mL^2}")
            st.caption("Energy eigenvalue")
    
    # Physics calculations
    x = np.linspace(0, L, 1000)
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    prob = psi**2
    E = (n**2 * np.pi**2) / (2 * L**2)
    
    # Calculate proper y-axis range
    psi_max = np.max(np.abs(psi)) * 1.2
    prob_max = np.max(prob) * 1.2
    y_max = max(psi_max, prob_max)
    y_min = -psi_max
    
    # Create the plot
    fig1 = go.Figure()
    
    # Add box walls if enabled
    if show_box:
        # Left wall
        fig1.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max,
            line=dict(color="gray", width=4))
        # Right wall
        fig1.add_shape(type="line", x0=L, y0=y_min, x1=L, y1=y_max,
            line=dict(color="gray", width=4))
        # Box bottom
        fig1.add_shape(type="line", x0=0, y0=y_min, x1=L, y1=y_min,
            line=dict(color="gray", width=2))
        # Shaded regions outside
        fig1.add_shape(type="rect", x0=-1, y0=y_min, x1=0, y1=y_max,
            fillcolor="rgba(128,128,128,0.2)", line=dict(color="rgba(0,0,0,0)"))
        fig1.add_shape(type="rect", x0=L, y0=y_min, x1=L+1, y1=y_max,
            fillcolor="rgba(128,128,128,0.2)", line=dict(color="rgba(0,0,0,0)"))
    
    # Add wavefunction and probability
    fig1.add_trace(go.Scatter(x=x, y=psi, mode='lines', name=f'ψ_{n}(x)',
        line=dict(color='blue', width=3)))
    fig1.add_trace(go.Scatter(x=x, y=prob, mode='lines', name=f'|ψ_{n}(x)|²',
        line=dict(color='red', width=3)))
    
    # Zero line
    fig1.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    
    # Style
    fig1.update_layout(
        title=dict(text=f"Quantum State n={n}, Energy E={E:.3f}", font=dict(size=20)),
        xaxis=dict(title="Position (x)", range=[-0.5, L+0.5]),
        yaxis=dict(title="Amplitude", range=[y_min, y_max]),
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Energy Level", f"n = {n}")
    with col2:
        st.metric("Energy", f"E_{n} = {E:.3f}", help="In units where ħ²/2m = 1")
    with col3:
        st.metric("Wavelength", f"λ = {2*L/n:.3f}")

# ============================================
# TAB 2: QUANTUM TUNNELING
# ============================================
with tab2:
    st.header("Quantum Tunneling Through a Barrier")
    st.write("A quantum particle encountering a potential barrier - demonstrating wave-particle duality")
    
    # Controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        E = st.slider("Particle Energy (E)", 0.1, 3.0, 1.0, 0.1,
                      help="Total energy of the incoming particle")
    with col2:
        V0 = st.slider("Barrier Height (V₀)", 0.1, 3.0, 1.5, 0.1,
                       help="Height of the potential barrier")
    with col3:
        a = st.slider("Barrier Width (a)", 0.5, 5.0, 2.0, 0.1,
                      help="Width of the potential barrier")
    
    # Show tunneling equations and derivation
    with st.expander("📐 Mathematical Derivation", expanded=True):
        st.markdown("""
        ### Time-Independent Schrödinger Equation
        
        For a particle of mass m encountering a rectangular barrier:
        """)
        
        st.latex(r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi")
        
        st.markdown("""
        Where the potential is:
        - V(x) = 0 for x < 0 (Region I)
        - V(x) = V₀ for 0 ≤ x ≤ a (Region II)  
        - V(x) = 0 for x > a (Region III)
        
        ### Solutions in Each Region
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Region I & III:** V = 0")
            st.latex(r"k_1 = \sqrt{\frac{2mE}{\hbar^2}}")
            st.latex(r"\psi_I = e^{ik_1x} + Re^{-ik_1x}")
            st.latex(r"\psi_{III} = Te^{ik_1x}")
        
        with col2:
            st.markdown("**Region II:** V = V₀")
            if E < V0:
                st.latex(r"\kappa = \sqrt{\frac{2m(V_0-E)}{\hbar^2}}")
                st.latex(r"\psi_{II} = Ae^{\kappa x} + Be^{-\kappa x}")
            else:
                st.latex(r"k_2 = \sqrt{\frac{2m(E-V_0)}{\hbar^2}}")
                st.latex(r"\psi_{II} = Ce^{ik_2x} + De^{-ik_2x}")
        
        st.markdown("""
        ### Transmission Coefficient
        
        By matching boundary conditions at x = 0 and x = a (continuity of ψ and dψ/dx):
        """)
        
        if E < V0:
            st.markdown("**Tunneling Case (E < V₀):**")
            st.latex(r"T = \frac{|T|^2}{|1|^2} = \frac{1}{1 + \frac{V_0^2 \sinh^2(\kappa a)}{4E(V_0-E)}}")
            st.markdown("For thick barriers (κa >> 1):")
            st.latex(r"T \approx \frac{16E(V_0-E)}{V_0^2}e^{-2\kappa a}")
        else:
            st.markdown("**Above Barrier Case (E > V₀):**")
            st.latex(r"T = \frac{1}{1 + \frac{V_0^2 \sin^2(k_2 a)}{4E(E-V_0)}}")
            st.markdown("Notice: T = 1 when sin(k₂a) = 0, giving resonant transmission!")
    
    # Physics calculations
    x = np.linspace(-8, 12, 2000)
    
    # Potential barrier
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= a)] = V0
    
    # Wave vectors
    k1 = np.sqrt(2*E)  # Outside barrier
    
    # Calculate transmission coefficient
    if E < V0:
        # Tunneling regime
        kappa = np.sqrt(2*(V0-E))
        T = 1 / (1 + (V0**2 * np.sinh(kappa*a)**2) / (4*E*(V0-E)))
        k2 = 1j * kappa  # Imaginary wave vector inside barrier
    else:
        # Above barrier
        k2 = np.sqrt(2*(E-V0))
        T = 1 / (1 + (V0**2 * np.sin(k2*a)**2) / (4*E*(E-V0)))
    
    R = 1 - T  # Reflection coefficient
    
    # Wavefunction (showing real part for visualization)
    psi = np.zeros_like(x, dtype=complex)
    psi_prob = np.zeros_like(x)
    
    # Region I: x < 0 (incident + reflected wave)
    mask1 = x < 0
    psi[mask1] = np.exp(1j*k1*x[mask1]) + np.sqrt(R)*np.exp(-1j*k1*x[mask1])
    
    # Region II: 0 <= x <= a (inside barrier)
    mask2 = (x >= 0) & (x <= a)
    if E < V0:
        # Exponential decay
        A = (1 + np.sqrt(R)) / (2*np.cosh(kappa*a/2))
        psi[mask2] = A * np.exp(-kappa*(x[mask2]-a/2))
    else:
        # Oscillatory
        psi[mask2] = 2*np.cos(k2*(x[mask2]-a/2)) / np.cos(k2*a/2)
    
    # Region III: x > a (transmitted wave)
    mask3 = x > a
    psi[mask3] = np.sqrt(T) * np.exp(1j*k1*x[mask3])
    
    # Calculate probability density
    psi_prob = np.abs(psi)**2
    
    # Calculate proper y-axis range
    psi_real_max = np.max(np.abs(np.real(psi))) * 1.2
    prob_max = np.max(psi_prob) * 1.2
    pot_max = V0 * 1.2
    y_max = max(psi_real_max, prob_max, pot_max, E + 0.5)
    y_min = -psi_real_max
        # Create plot
    fig2 = go.Figure()
    
    # Potential barrier
    fig2.add_trace(go.Scatter(x=x, y=V, name="Potential V(x)",
                             line=dict(color='black', width=3),
                             fill='tozeroy', fillcolor='rgba(128,128,128,0.2)'))
    
    # Energy level
    fig2.add_hline(y=E, line_dash="dash", line_color="green", line_width=2,
                   annotation_text=f"E = {E:.2f}")
    
    # Wavefunction (real part)
    fig2.add_trace(go.Scatter(x=x, y=np.real(psi), name="Re(ψ)",
                             line=dict(color='blue', width=2)))
    
    # Probability density
    fig2.add_trace(go.Scatter(x=x, y=psi_prob, name="|ψ|²",
                             line=dict(color='red', width=2)))
    
    # Style with proper y-axis range
    fig2.update_layout(
        title=f"Quantum Tunneling - E/V₀ = {E/V0:.2f}",
        xaxis=dict(title="Position (x)", range=[-8, 12]),
        yaxis=dict(title="Energy / Amplitude", range=[y_min, y_max]),
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Results metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transmission", f"{T:.1%}",
                  delta=f"{T*100:.1f}% quantum" if E < V0 else None,
                  help="Probability of passing through barrier")
    
    with col2:
        st.metric("Reflection", f"{R:.1%}",
                  help="Probability of being reflected")
    
    with col3:
        classical = "0%" if E < V0 else "100%"
        st.metric("Classical Result", classical,
                  help="What classical mechanics predicts")
    
    with col4:
        regime = "Tunneling" if E < V0 else "Above Barrier"
        st.metric("Regime", regime,
                  delta="E < V₀" if E < V0 else "E > V₀")
    
    # Educational info
    with st.expander("📚 Physical Interpretation and Applications"):
        st.markdown(f"""
        ### Current Setup
        - **Particle Energy**: E = {E:.2f}
        - **Barrier Height**: V₀ = {V0:.2f}
        - **Barrier Width**: a = {a:.2f}
        - **Energy Ratio**: E/V₀ = {E/V0:.2f}
        
        ### Physical Interpretation
        
        {"#### ⚡ Tunneling Regime (E < V₀)" if E < V0 else "#### 🌊 Above-Barrier Regime (E > V₀)"}
        
        """)
        
        if E < V0:
            st.markdown(f"""
            The particle has **less energy than the barrier height**, yet there's a **{T:.1%} chance** 
            it will appear on the other side! This is purely quantum mechanical - classically impossible.
            
            - **Wavefunction decay**: The wavefunction decays exponentially inside the barrier
            - **Decay length**: 1/κ = {1/kappa:.2f} (characteristic penetration depth)
            - **Width dependence**: Transmission ∝ exp(-2κa) for thick barriers
            - **Energy dependence**: Higher E → larger transmission
            """)
        else:
            st.markdown(f"""
            The particle has **more energy than the barrier**, so classically it should always pass.
            But quantum mechanically, there's still a **{R:.1%} chance of reflection**!
            
            - **Wave interference**: Partial waves reflected at x=0 and x=a interfere
            - **Resonances**: When k₂a = nπ, we get T = 100% (perfect transmission)
            - **Current k₂a = {k2*a:.2f}** (nearest resonance at {np.round(k2*a/np.pi)*np.pi:.2f})
            """)
        
        st.markdown("""
        ### Real-World Applications of Quantum Tunneling
        
        1. **Scanning Tunneling Microscope (STM)**
           - Electrons tunnel between tip and surface
           - Current ∝ exp(-2κd) where d is tip-surface distance
           - Achieves atomic resolution imaging
        
        2. **Nuclear Fusion in Stars**
           - Protons tunnel through Coulomb barrier
           - Without tunneling, stars couldn't shine!
           - Sun's core temperature too low for classical fusion
        
        3. **Radioactive Alpha Decay**
           - Alpha particles tunnel out of nuclear potential
           - Explains Geiger-Nuttall law for decay rates
        
        4. **Semiconductor Devices**
           - Tunnel diodes: negative differential resistance
           - Flash memory: electrons tunnel through oxide barrier
           - Josephson junctions in quantum computers
        
        5. **Chemical Reactions**
           - Proton transfer in DNA base pairs
           - Enzyme catalysis enhancement
           - Quantum biology effects
        """)

# Add "Learn More" section with proper derivation
# ============================================
# TAB 3: QUANTUM HARMONIC OSCILLATOR
# ============================================
with tab3:
    st.header("Quantum Harmonic Oscillator")
    st.write("Visualizing quantum states in parabolic potentials - the most important exactly solvable system")
    
    # Choose 1D or 2D
    oscillator_dim = st.radio("Select dimension:", ["1D Oscillator", "2D Oscillator"], horizontal=True)
    
    if oscillator_dim == "1D Oscillator":
        # 1D Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n = st.slider("Quantum number (n)", 0, 10, 0,
                         help="n = 0 is the ground state")
        with col2:
            omega = st.slider("Frequency (ω)", 0.5, 2.0, 1.0, 0.1,
                             help="Oscillator frequency")
        with col3:
            show_classical = st.checkbox("Show classical turning points", value=True)
            show_probability = st.checkbox("Show |ψ|²", value=True)
        
        # Show equations
        with st.expander("📐 Mathematical Description", expanded=True):
            st.markdown("### The Quantum Harmonic Oscillator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Potential Energy:**")
                st.latex(r"V(x) = \frac{1}{2}m\omega^2 x^2")
                
                st.markdown("**Schrödinger Equation:**")
                st.latex(r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + \frac{1}{2}m\omega^2 x^2\psi = E\psi")
            
            with col2:
                st.markdown("**Energy Eigenvalues:**")
                st.latex(r"E_n = \hbar\omega\left(n + \frac{1}{2}\right)")
                
                st.markdown("**Ground State Energy:**")
                st.latex(r"E_0 = \frac{1}{2}\hbar\omega")
            
            st.markdown("### Wavefunctions")
            st.latex(r"\psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}} H_n\left(\sqrt{\frac{m\omega}{\hbar}}x\right) \exp\left(-\frac{m\omega x^2}{2\hbar}\right)")
            
            st.markdown("""
            Where:
            - H_n are Hermite polynomials
            - The characteristic length scale is: x₀ = √(ℏ/mω)
            - Classical turning points at: x = ±√(2E/mω²)
            """)
        
        # Physics calculations (using natural units where ℏ = m = 1)
        x0 = np.sqrt(1/(omega))  # characteristic length
        E_n = omega * (n + 0.5)
        
        # Classical turning points
        x_turn = np.sqrt(2*E_n/omega)
        
        # Grid
        x_max = max(3*x0, x_turn + x0)
        x = np.linspace(-x_max, x_max, 1000)
        
        # Potential
        V = 0.5 * omega**2 * x**2
        
        # Hermite polynomial and wavefunction
        H_n = hermite(n)
        prefactor = (omega/np.pi)**0.25 / np.sqrt(2**n * factorial(n))
        xi = np.sqrt(omega) * x
        psi = prefactor * H_n(xi) * np.exp(-xi**2 / 2)
        psi_squared = psi**2
        
        # Normalize for display (scale to fit nicely with potential)
        psi_display = psi * 0.5 * E_n / np.max(np.abs(psi)) + E_n
        psi_squared_display = psi_squared * 0.5 * E_n / np.max(psi_squared) + E_n
        
        # Create plot
        fig_1d = go.Figure()
        
        # Potential
        fig_1d.add_trace(go.Scatter(x=x, y=V, name="V(x) = ½mω²x²",
                                   line=dict(color='black', width=3)))
        
        # Energy level
        fig_1d.add_hline(y=E_n, line_dash="dash", line_color="green",
                        annotation_text=f"E_{n} = {E_n:.2f}")
        
        # Classical turning points
        if show_classical:
            fig_1d.add_vline(x=-x_turn, line_dash="dot", line_color="gray",
                            annotation_text="Classical turning point")
            fig_1d.add_vline(x=x_turn, line_dash="dot", line_color="gray")
            
            # Shaded classically forbidden region
            fig_1d.add_shape(type="rect", x0=-x_max, y0=0, x1=-x_turn, y1=E_n,
                            fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
            fig_1d.add_shape(type="rect", x0=x_turn, y0=0, x1=x_max, y1=E_n,
                            fillcolor="rgba(255,0,0,0.1)", line=dict(width=0))
        
        # Wavefunction (shifted to energy level)
        fig_1d.add_trace(go.Scatter(x=x, y=psi_display, name=f"ψ_{n}(x)",
                                   line=dict(color='blue', width=2)))
        
        # Probability density
        if show_probability:
            fig_1d.add_trace(go.Scatter(x=x, y=psi_squared_display, name=f"|ψ_{n}(x)|²",
                                       line=dict(color='red', width=2), fill='tonexty'))
        
        # Style
        fig_1d.update_layout(
            title=f"1D Quantum Harmonic Oscillator - n = {n}",
            xaxis=dict(title="Position (x)", range=[-x_max, x_max]),
            yaxis=dict(title="Energy", range=[0, max(V.max(), E_n + 1)]),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_1d, use_container_width=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Energy", f"E_{n} = {E_n:.3f}")
        with col2:
            st.metric("Zero-point energy", f"E₀ = {omega/2:.3f}")
        with col3:
            st.metric("Level spacing", f"ΔE = {omega:.3f}")
        with col4:
            st.metric("Classical extent", f"±{x_turn:.3f}")
    
    else:  # 2D Oscillator
        st.markdown("### 2D Isotropic Harmonic Oscillator")
        st.write("Independent oscillators in x and y directions: ψ(x,y) = ψₙₓ(x) × ψₙᵧ(y)")
        
        # 2D Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            nx = st.slider("Quantum number nₓ", 0, 5, 0)
        with col2:
            ny = st.slider("Quantum number nᵧ", 0, 5, 0)
        with col3:
            omega_2d = st.slider("Frequency ω", 0.5, 2.0, 1.0, 0.1)
        with col4:
            plot_type = st.selectbox("Visualization", 
                                     ["3D Surface", "2D Heatmap", "Contour Plot"])
        
        # Show equations for 2D
        with st.expander("📐 2D Harmonic Oscillator Theory"):
            st.markdown("""
            ### Separable Solution
            
            For V(x,y) = ½mω²(x² + y²), the wavefunction separates:
            """)
            
            st.latex(r"\psi_{n_x,n_y}(x,y) = \psi_{n_x}(x) \cdot \psi_{n_y}(y)")
            
            st.latex(r"E_{n_x,n_y} = \hbar\omega(n_x + n_y + 1)")
            
            st.markdown("""
            ### Degeneracy
            
            The energy depends on N = nₓ + nᵧ, so states with same N are degenerate.
            
            For example, E = 2ℏω can be achieved with:
            - (nₓ, nᵧ) = (1, 0)
            - (nₓ, nᵧ) = (0, 1)
            
            Degeneracy of level N is N + 1.
            """)
        
        # 2D Physics calculations
        x0_2d = np.sqrt(1/omega_2d)
        E_2d = omega_2d * (nx + ny + 1)
        
        # Create 2D grid
        extent = max(3*x0_2d, np.sqrt(2*E_2d/omega_2d) + x0_2d)
        x = np.linspace(-extent, extent, 100)
        y = np.linspace(-extent, extent, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate 2D wavefunction
        # First calculate 1D wavefunctions for x and y
        Hx = hermite(nx)
        Hy = hermite(ny)
        
        # Normalization and xi coordinates
        norm_x = (omega_2d/np.pi)**0.25 / np.sqrt(2**nx * factorial(nx))
        norm_y = (omega_2d/np.pi)**0.25 / np.sqrt(2**ny * factorial(ny))
        xi_x = np.sqrt(omega_2d) * X
        xi_y = np.sqrt(omega_2d) * Y
        
        # 2D wavefunction
        psi_x = norm_x * Hx(xi_x) * np.exp(-xi_x**2 / 2)
        psi_y = norm_y * Hy(xi_y) * np.exp(-xi_y**2 / 2)
        psi_2d = psi_x * psi_y
        prob_2d = np.abs(psi_2d)**2
        
        # Create visualization based on selection
        if plot_type == "3D Surface":
            fig_2d = go.Figure(data=[
                go.Surface(
                    x=x, y=y, z=prob_2d,
                    colorscale='Viridis',
                    name='|ψ(x,y)|²',
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                    )
                )
            ])
            
            fig_2d.update_layout(
                title=f"2D Harmonic Oscillator - |ψ_{nx},{ny}(x,y)|²",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="|ψ|²",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600
            )
        
        elif plot_type == "2D Heatmap":
            fig_2d = go.Figure(data=go.Heatmap(
                x=x, y=y, z=prob_2d,
                colorscale='Viridis',
                colorbar=dict(title="|ψ|²")
            ))
            
            fig_2d.update_layout(
                title=f"2D Harmonic Oscillator - Probability Density (nₓ={nx}, nᵧ={ny})",
                xaxis_title="x",
                yaxis_title="y",
                height=600,
                width=700,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
        
        else:  # Contour Plot
            fig_2d = go.Figure(data=go.Contour(
                x=x, y=y, z=prob_2d,
                colorscale='Viridis',
                contours=dict(showlabels=True, labelfont=dict(size=12)),
                colorbar=dict(title="|ψ|²")
            ))
            
            # Add nodal lines (where psi = 0, shown as white contours)
            fig_2d.add_trace(go.Contour(
                x=x, y=y, z=psi_2d,
                showscale=False,
                contours=dict(
                    start=0, end=0, size=0.1,
                    coloring='lines',
                    showlabels=False
                ),
                line=dict(color='white', width=2),
                name='Nodal lines'
            ))
            
            fig_2d.update_layout(
                title=f"2D Harmonic Oscillator - Contour Plot (nₓ={nx}, nᵧ={ny})",
                xaxis_title="x",
                yaxis_title="y",
                height=600,
                width=700,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
        
        st.plotly_chart(fig_2d, use_container_width=True)
        
        # 2D Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Energy", f"E = {E_2d:.3f}ℏω")
        with col2:
            st.metric("Total quantum number", f"N = {nx + ny}")
        with col3:
            st.metric("Degeneracy", f"{nx + ny + 1} states at this energy")
    
    # Educational content for harmonic oscillator
    with st.expander("📚 Physical Insights and Applications"):
        st.markdown("""
        ### Why is the Harmonic Oscillator So Important?
        
        1. **Ubiquitous in Physics**:
           - Small oscillations around equilibrium are harmonic
           - Applies to molecules, atoms in crystals, electromagnetic fields
           - Foundation of quantum field theory (creation/annihilation operators)
        
        2. **Exactly Solvable**:
           - One of few systems with exact analytical solutions
           - Solutions involve Hermite polynomials
           - Energy levels are equally spaced
        
        3. **Zero-Point Energy**:
           - Ground state energy E₀ = ½ℏω ≠ 0
           - Consequence of uncertainty principle
           - Particle can never be at rest at x = 0
        
        4. **Classical Limit**:
           - For large n, probability density approaches classical distribution
           - Time spent at position x ∝ 1/|velocity(x)|
           - Demonstrates correspondence principle
        
        ### Applications:
        
        **Molecular Vibrations**:
        - Diatomic molecules vibrate like quantum harmonic oscillators
        - Infrared spectroscopy measures transitions: ΔE = ℏω
        - Selection rule: Δn = ±1 for dipole transitions
        
        **Quantum Optics**:
        - Electromagnetic field modes are harmonic oscillators
        - Photon number states |n⟩ correspond to energy levels
        - Coherent states are displaced ground states
        
        **Solid State Physics**:
        - Phonons: quantized lattice vibrations
        - Einstein model: N independent oscillators
        - Explains heat capacity at low temperatures
        
        **Quantum Computing**:
        - Trapped ions in harmonic potentials
        - Vibrational states used as qubits
        - Sideband cooling to ground state
        
        ### Mathematical Beauty:
        
        The ladder operators â and â† connect different energy levels:
        - â|n⟩ = √n|n-1⟩ (lowering operator)
        - â†|n⟩ = √(n+1)|n+1⟩ (raising operator)
        - [â, â†] = 1 (canonical commutation relation)
        """)

with st.sidebar:
    st.markdown("### 🎓 About This Visualizer")
    st.markdown("""
    This app visualizes fundamental quantum mechanical systems:
    
    - **Particle in a Box**: Energy quantization in confined systems
    - **Quantum Tunneling**: Particles passing through classically forbidden barriers
    
    More quantum systems coming soon!
    """)
    
    st.markdown("### 📚 References")
    st.markdown("""
    **Textbooks:**
    - Griffiths, D.J. *Introduction to Quantum Mechanics* (3rd ed.)
    - Shankar, R. *Principles of Quantum Mechanics* (2nd ed.)
    
    **Interactive Resources:**
    - [PhET Quantum Simulations](https://phet.colorado.edu/en/simulations/quantum-tunneling)
    """)
    
    st.markdown("### 🔢 Units")
    st.markdown("""
    This simulation uses natural units where:
    - ℏ = 1 (reduced Planck constant)
    - 2m = 1 (twice the particle mass)
    - All energies in units of ℏ²/2m
    - All lengths in units of √(ℏ/2mE₀)
    """)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit | Quantum Mechanics Visualization Suite")