"""
Rectangular Reinforced Concrete Column Design Application
Based on Thai Engineering Standards (ACI 318 Metric)
"""

import streamlit as st
import numpy as np # type: ignore
import math
import tempfile
import os
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

# PDF generation
from fpdf import FPDF

# =============================================================================
# UNIT CONVERSION CONSTANTS
# =============================================================================
# Convert Thai common units to SI (N, mm)

# Length: cm to mm
CM_TO_MM = 10.0

# Force: Ton to Newton (1 ton = 1000 kg, 1 kgf = 9.80665 N)
TON_TO_N = 1000.0 * 9.80665  # 9806.65 N

# Stress: ksc (kg/cm²) to MPa (N/mm²)
# 1 ksc = 1 kgf/cm² = 9.80665 N / (10 mm)² = 0.0980665 N/mm² = 0.0980665 MPa
KSC_TO_MPA = 0.0980665

# MPa to N/mm² (same unit, just for clarity)
MPA_TO_NMMSQ = 1.0

# =============================================================================
# MATERIAL DATA
# =============================================================================

# Steel grades according to Thai Industrial Standards (TIS)
STEEL_GRADES = {
    "SD30": {"fy_mpa": 295, "description": "SD30 (fy = 295 MPa)"},
    "SD40": {"fy_mpa": 390, "description": "SD40 (fy = 390 MPa)"},
    "SD50": {"fy_mpa": 490, "description": "SD50 (fy = 490 MPa)"},
}

# Stirrup bar options (Round Bars - RB/SR24)
STIRRUP_BARS = {
    "RB6": {"diameter": 6, "fy_mpa": 235, "description": "RB6 - SR24 (fy = 235 MPa ≈ 2400 ksc)"},
    "RB9": {"diameter": 9, "fy_mpa": 235, "description": "RB9 - SR24 (fy = 235 MPa ≈ 2400 ksc)"},
}

# Rebar diameters (Thai standard deformed bars)
REBAR_DIAMETERS = {
    "DB12": 12,  # mm
    "DB16": 16,  # mm
    "DB20": 20,  # mm
    "DB25": 25,  # mm
    "DB28": 28,  # mm
    "DB32": 32,  # mm
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_fc_ksc_to_mpa(fc_ksc: float) -> float:
    """Convert concrete strength from ksc to MPa"""
    return fc_ksc * KSC_TO_MPA


def convert_dimension_cm_to_mm(dim_cm: float) -> float:
    """Convert dimension from cm to mm"""
    return dim_cm * CM_TO_MM


def convert_force_ton_to_n(force_ton: float) -> float:
    """Convert force from Ton to Newton"""
    return force_ton * TON_TO_N


def get_rebar_area(diameter_mm: float) -> float:
    """Calculate single rebar area in mm²"""
    return math.pi * (diameter_mm ** 2) / 4


def calculate_total_steel_area(n_bars: int, diameter_mm: float) -> float:
    """Calculate total steel area in mm²"""
    return n_bars * get_rebar_area(diameter_mm)


# =============================================================================
# SLENDERNESS EFFECT CALCULATIONS (Moment Magnification Method)
# =============================================================================

def calculate_Ec(fc_mpa: float) -> float:
    """
    Calculate modulus of elasticity of concrete.
    Ec = 4700 * sqrt(f'c) in MPa (ACI 318)
    """
    return 4700 * math.sqrt(fc_mpa)


def calculate_slenderness_ratio(k: float, Lu_mm: float, h_mm: float) -> float:
    """
    Calculate slenderness ratio kLu/r.
    r = 0.3h for rectangular sections
    """
    r = 0.3 * h_mm  # Radius of gyration
    return (k * Lu_mm) / r


def calculate_critical_buckling_load(
    Ec_mpa: float,
    b_mm: float,
    h_mm: float,
    k: float,
    Lu_mm: float,
    beta_dns: float = 0.6
) -> float:
    """
    Calculate critical buckling load Pc using ACI 318 moment magnification method.
    
    EI = (0.4 * Ec * Ig) / (1 + βdns)
    Pc = π² * EI / (k * Lu)²
    
    Args:
        Ec_mpa: Modulus of elasticity of concrete (MPa)
        b_mm: Column width (mm)
        h_mm: Column depth (mm)
        k: Effective length factor
        Lu_mm: Unsupported length (mm)
        beta_dns: Sustained load factor (default 0.6)
    
    Returns:
        Critical buckling load Pc in N
    """
    # Moment of inertia (gross section)
    Ig = b_mm * (h_mm ** 3) / 12  # mm⁴
    
    # Flexural stiffness
    EI = (0.4 * Ec_mpa * Ig) / (1 + beta_dns)  # N·mm²
    
    # Critical buckling load
    kLu = k * Lu_mm
    Pc = (math.pi ** 2) * EI / (kLu ** 2)  # N
    
    return Pc


def calculate_moment_magnification_factor(
    Pu_N: float,
    Pc_N: float,
    Cm: float = 1.0
) -> float:
    """
    Calculate moment magnification factor δns for non-sway frames.
    
    δns = Cm / (1 - Pu / (0.75 * Pc))
    
    Args:
        Pu_N: Factored axial load (N)
        Pc_N: Critical buckling load (N)
        Cm: Equivalent uniform moment factor (default 1.0, conservative)
    
    Returns:
        Moment magnification factor δns (≥ 1.0)
    """
    # Avoid division by zero or negative denominator
    denominator = 1 - Pu_N / (0.75 * Pc_N)
    
    if denominator <= 0:
        # Column is unstable
        return float('inf')
    
    delta_ns = Cm / denominator
    
    # δns must be at least 1.0
    return max(delta_ns, 1.0)


def check_slenderness_limit(slenderness_ratio: float, limit: float = 22.0) -> bool:
    """
    Check if column is classified as slender (long column).
    
    Per ACI 318 simplified method:
    - If kLu/r > 22, column is slender and requires moment magnification
    
    Returns:
        True if column is slender (long column)
    """
    return slenderness_ratio > limit


# =============================================================================
# COLUMN CROSS-SECTION VISUALIZATION
# =============================================================================

def plot_column_section(
    b_mm: float,
    h_mm: float,
    cover_mm: float,
    main_bar_dia_mm: float,
    stirrup_dia_mm: float,
    n_bars_x: int,
    n_bars_y: int,
    stirrup_spacing_m: float = 0.15,
    stirrup_bar_name: str = "RB9",
    main_bar_name: str = "DB20"
):
    """
    Create a technical drawing of the column cross-section using Matplotlib.
    
    Args:
        b_mm: Column width (mm)
        h_mm: Column depth (mm)
        cover_mm: Clear cover to stirrup (mm)
        main_bar_dia_mm: Main bar diameter (mm)
        stirrup_dia_mm: Stirrup diameter (mm)
        n_bars_x: Number of bars along width (b)
        n_bars_y: Number of bars along depth (h)
        stirrup_spacing_m: Stirrup spacing (m)
        stirrup_bar_name: Stirrup bar designation (e.g., "RB9")
        main_bar_name: Main bar designation (e.g., "DB20")
    
    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    
    # Convert to cm for display
    b_cm = b_mm / 10
    h_cm = h_mm / 10
    
    # Create figure with technical drawing style
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))
    
    # Set background to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Drawing scale: work in mm, then convert display
    # Add margin for dimension lines
    margin = max(b_mm, h_mm) * 0.25
    
    # ==========================================================================
    # DRAW CONCRETE SECTION (Outer Rectangle)
    # ==========================================================================
    concrete_rect = Rectangle(
        (0, 0), b_mm, h_mm,
        linewidth=2.5,
        edgecolor='black',
        facecolor='#D3D3D3',  # Light gray for concrete
        zorder=1
    )
    ax.add_patch(concrete_rect)
    
    # Add concrete hatch pattern (optional - for technical look)
    # Using diagonal lines pattern
    for i in range(0, int(b_mm + h_mm), 15):
        x_start = max(0, i - h_mm)
        x_end = min(i, b_mm)
        y_start = max(0, h_mm - i)
        y_end = min(h_mm, h_mm - i + b_mm)
        if x_start < x_end:
            ax.plot([x_start, x_end], [i - x_start if i <= h_mm else h_mm - (i - h_mm) - x_start, 
                                        i - x_end if i <= h_mm else h_mm - (i - h_mm) - x_end],
                   'gray', linewidth=0.3, alpha=0.3, zorder=1.5)
    
    # ==========================================================================
    # DRAW STIRRUP (Rounded Rectangle inside cover)
    # ==========================================================================
    stirrup_offset = cover_mm + stirrup_dia_mm / 2  # To centerline of stirrup
    stirrup_width = b_mm - 2 * cover_mm - stirrup_dia_mm
    stirrup_height = h_mm - 2 * cover_mm - stirrup_dia_mm
    
    # Corner radius for stirrup bends (typically 2-4 x bar diameter)
    corner_radius = min(stirrup_dia_mm * 3, stirrup_width / 4, stirrup_height / 4)
    
    # Draw stirrup as rounded rectangle (using FancyBboxPatch)
    stirrup_rect = FancyBboxPatch(
        (cover_mm + stirrup_dia_mm / 2, cover_mm + stirrup_dia_mm / 2),
        stirrup_width, stirrup_height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        linewidth=stirrup_dia_mm * 0.4,  # Line width proportional to bar size
        edgecolor='#2E7D32',  # Dark green for stirrups
        facecolor='none',
        zorder=3
    )
    ax.add_patch(stirrup_rect)
    
    # ==========================================================================
    # CALCULATE REBAR POSITIONS
    # ==========================================================================
    # Distance from section edge to center of main bars
    bar_center_offset = cover_mm + stirrup_dia_mm + main_bar_dia_mm / 2
    
    # Available space for bar distribution
    available_width = b_mm - 2 * bar_center_offset
    available_height = h_mm - 2 * bar_center_offset
    
    # Calculate spacing
    if n_bars_x > 1:
        spacing_x = available_width / (n_bars_x - 1)
    else:
        spacing_x = 0
    
    if n_bars_y > 1:
        spacing_y = available_height / (n_bars_y - 1)
    else:
        spacing_y = 0
    
    # Generate bar positions (perimeter arrangement)
    bar_positions = []
    
    # Bottom row (all n_bars_x)
    for i in range(n_bars_x):
        x = bar_center_offset + i * spacing_x
        y = bar_center_offset
        bar_positions.append((x, y))
    
    # Top row (all n_bars_x)
    for i in range(n_bars_x):
        x = bar_center_offset + i * spacing_x
        y = h_mm - bar_center_offset
        bar_positions.append((x, y))
    
    # Left side (excluding corners - n_bars_y - 2)
    for j in range(1, n_bars_y - 1):
        x = bar_center_offset
        y = bar_center_offset + j * spacing_y
        bar_positions.append((x, y))
    
    # Right side (excluding corners - n_bars_y - 2)
    for j in range(1, n_bars_y - 1):
        x = b_mm - bar_center_offset
        y = bar_center_offset + j * spacing_y
        bar_positions.append((x, y))
    
    # ==========================================================================
    # DRAW MAIN REINFORCEMENT BARS
    # ==========================================================================
    for (x, y) in bar_positions:
        # Outer circle (bar outline)
        bar_circle = Circle(
            (x, y),
            radius=main_bar_dia_mm / 2,
            linewidth=1.5,
            edgecolor='black',
            facecolor='#1565C0',  # Blue for main bars
            zorder=5
        )
        ax.add_patch(bar_circle)
        
        # Inner detail (cross pattern for bar end view)
        cross_size = main_bar_dia_mm * 0.25
        ax.plot([x - cross_size, x + cross_size], [y, y], 
               color='white', linewidth=1, zorder=6)
        ax.plot([x, x], [y - cross_size, y + cross_size], 
               color='white', linewidth=1, zorder=6)
    
    # ==========================================================================
    # DIMENSION LINES AND ANNOTATIONS
    # ==========================================================================
    dim_offset = margin * 0.4  # Offset for dimension lines
    arrow_props = dict(arrowstyle='<->', color='black', lw=1.2)
    
    # --- Width dimension (b) at bottom ---
    y_dim_b = -dim_offset
    ax.annotate('', xy=(0, y_dim_b), xytext=(b_mm, y_dim_b),
               arrowprops=arrow_props)
    # Dimension extension lines
    ax.plot([0, 0], [0, y_dim_b - 5], 'k-', linewidth=0.8)
    ax.plot([b_mm, b_mm], [0, y_dim_b - 5], 'k-', linewidth=0.8)
    # Dimension text
    ax.text(b_mm / 2, y_dim_b - dim_offset * 0.3, f'b = {b_cm:.0f} cm',
           ha='center', va='top', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
    
    # --- Height dimension (h) at right ---
    x_dim_h = b_mm + dim_offset
    ax.annotate('', xy=(x_dim_h, 0), xytext=(x_dim_h, h_mm),
               arrowprops=arrow_props)
    # Dimension extension lines
    ax.plot([b_mm, x_dim_h + 5], [0, 0], 'k-', linewidth=0.8)
    ax.plot([b_mm, x_dim_h + 5], [h_mm, h_mm], 'k-', linewidth=0.8)
    # Dimension text
    ax.text(x_dim_h + dim_offset * 0.3, h_mm / 2, f'h = {h_cm:.0f} cm',
           ha='left', va='center', fontsize=11, fontweight='bold', rotation=90,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))
    
    # --- Cover dimension ---
    # Show cover at top-left corner
    cover_x = cover_mm / 2
    cover_y = h_mm - cover_mm / 2
    ax.annotate('', xy=(0, h_mm), xytext=(cover_mm, h_mm - cover_mm),
               arrowprops=dict(arrowstyle='<->', color='#666666', lw=0.8))
    ax.text(-dim_offset * 0.3, h_mm - cover_mm / 2, f'c = {cover_mm / 10:.1f} cm',
           ha='right', va='center', fontsize=9, color='#666666',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))
    
    # ==========================================================================
    # ANNOTATION ARROWS FOR BARS AND STIRRUPS
    # ==========================================================================
    total_bars = len(bar_positions)
    
    # --- Main bar annotation (pointing to a corner bar) ---
    # Point to top-right corner bar
    corner_bar_x = b_mm - bar_center_offset
    corner_bar_y = h_mm - bar_center_offset
    
    ax.annotate(
        f'{total_bars}-{main_bar_name}',
        xy=(corner_bar_x, corner_bar_y),
        xytext=(b_mm + dim_offset * 1.5, h_mm + dim_offset * 0.5),
        fontsize=11, fontweight='bold', color='#1565C0',
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', 
                 edgecolor='#1565C0', linewidth=1.5),
        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5,
                       connectionstyle='arc3,rad=-0.2')
    )
    
    # --- Stirrup annotation (pointing to stirrup at top) ---
    stirrup_label_x = b_mm / 2
    stirrup_label_y = h_mm - cover_mm - stirrup_dia_mm / 2
    
    ax.annotate(
        f'Stirrup {stirrup_bar_name}\n@ {stirrup_spacing_m:.2f} m',
        xy=(stirrup_label_x, stirrup_label_y),
        xytext=(-dim_offset * 0.8, h_mm + dim_offset * 0.5),
        fontsize=10, fontweight='bold', color='#2E7D32',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', 
                 edgecolor='#2E7D32', linewidth=1.5),
        arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5,
                       connectionstyle='arc3,rad=0.2')
    )
    
    # ==========================================================================
    # TITLE AND FINISHING
    # ==========================================================================
    ax.set_title('Column Cross-Section', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set equal aspect ratio for correct proportions
    ax.set_aspect('equal')
    
    # Set axis limits with margin for annotations
    ax.set_xlim(-margin, b_mm + margin * 1.2)
    ax.set_ylim(-margin * 0.8, h_mm + margin * 0.8)
    
    # Remove axis ticks and labels (technical drawing style)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add scale indicator at bottom
    ax.text(0.5, -0.08, f'Scale: Drawing not to scale | Units: mm',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=9, color='gray', style='italic')
    
    plt.tight_layout()
    
    return fig


# =============================================================================
# PDF REPORT GENERATION - 2-Page A4 Calculation Report
# =============================================================================

def create_pdf_report(
    design_inputs: dict,
    results: dict,
    pm_results: dict | None = None,
    fig_interaction=None,
    fig_section=None,
    font_path=None
) -> bytes:
    """
    Create a professional 2-page A4 PDF calculation report for RC Column Design.
    
    Page 1: Design Data, Cross-Section, Initial Calculations
    Page 2: P-M Interaction Diagram, Capacity Check, Conclusion
    
    Args:
        design_inputs: Dictionary containing design input parameters
        results: Dictionary containing analysis results
        pm_results: Dictionary containing P-M interaction curve data
        fig_interaction: Matplotlib figure for P-M interaction diagram
        fig_section: Matplotlib figure for column cross-section
        font_path: Optional path to Thai-compatible TTF font
    
    Returns:
        PDF file as bytes
    """
    
    # Initialize PDF (A4 size: 210 x 297 mm)
    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=False)
    
    # Extract all design inputs
    fc_ksc = design_inputs.get('fc_ksc', 240)
    fc_mpa = design_inputs.get('fc_mpa', 24)
    fy_mpa = design_inputs.get('fy_mpa', 390)
    steel_grade = design_inputs.get('steel_grade', 'SD40')
    
    b_cm = design_inputs.get('b_cm', 30)
    h_cm = design_inputs.get('h_cm', 40)
    b_mm = b_cm * 10
    h_mm = h_cm * 10
    cover_cm = design_inputs.get('cover_cm', 4)
    Ag_cm2 = design_inputs.get('Ag_cm2', b_cm * h_cm)
    Ag_mm2 = Ag_cm2 * 100
    
    main_bar = design_inputs.get('main_bar', 'DB20')
    n_bars_x = design_inputs.get('n_bars_x', 3)
    n_bars_y = design_inputs.get('n_bars_y', 3)
    total_bars = design_inputs.get('total_bars', 8)
    As_cm2 = design_inputs.get('As_cm2', 25)
    As_mm2 = As_cm2 * 100
    rho_percent = design_inputs.get('rho_percent', 2.0)
    
    stirrup_bar = design_inputs.get('stirrup_bar', 'RB9')
    stirrup_spacing_m = design_inputs.get('stirrup_spacing_m', 0.15)
    
    Pu_ton = design_inputs.get('Pu_ton', 100)
    Mu_tonm = design_inputs.get('Mu_tonm', 10)
    
    consider_slenderness = design_inputs.get('consider_slenderness', False)
    Lu_m = design_inputs.get('Lu_m', 3.0)
    k_factor = design_inputs.get('k_factor', 1.0)
    slenderness_ratio = design_inputs.get('slenderness_ratio', 0)
    is_long_column = design_inputs.get('is_long_column', False)
    Pc_N = design_inputs.get('Pc_N', 0)
    Pc_ton = Pc_N / 9806.65 if Pc_N > 0 else 0
    delta_ns = design_inputs.get('delta_ns', 1.0)
    Mc_tonm = design_inputs.get('Mc_tonm', Mu_tonm)
    beta_dns = design_inputs.get('beta_dns', 0.6)
    Cm = design_inputs.get('Cm', 1.0)
    
    # Results
    Pn_max_ton = results.get('Pn_max_ton', 0)
    phi_Pn_max_ton = results.get('phi_Pn_max_ton', 0)
    Mn_max_tonm = results.get('Mn_max_tonm', 0)
    phi_Mn_max_tonm = results.get('phi_Mn_max_tonm', 0)
    status = results.get('status', 'UNKNOWN')
    dc_ratio = results.get('dc_ratio', 0)
    is_safe = 'SAFE' in status.upper() or 'OK' in status.upper()
    
    # Temporary files for images
    temp_files = []
    
    try:
        # =====================================================================
        # PAGE 1: DESIGN DATA & INITIAL CALCULATIONS
        # =====================================================================
        pdf.add_page()
        
        # ----- HEADER -----
        pdf.set_fill_color(0, 51, 102)  # Dark blue
        pdf.rect(0, 0, 210, 25, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 18)
        pdf.set_xy(10, 8)
        pdf.cell(0, 8, 'RC COLUMN DESIGN CALCULATION SHEET', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, 'Based on ACI 318 / Thai Engineering Standards (EIT)', new_x="LMARGIN", new_y="NEXT", align='C')
        
        # Reset text color
        pdf.set_text_color(0, 0, 0)
        
        # ----- Project Info Bar -----
        pdf.set_xy(10, 28)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(63, 6, f'Date: {datetime.now().strftime("%Y-%m-%d")}', border=1, new_x="RIGHT", new_y="TOP", fill=True)
        pdf.cell(63, 6, f'Project: RC Column Design', border=1, new_x="RIGHT", new_y="TOP", fill=True)
        pdf.cell(64, 6, f'Page: 1 of 2', border=1, new_x="LMARGIN", new_y="NEXT", fill=True, align='R')
        
        # ----- SECTION 1: DESIGN INPUTS TABLE -----
        pdf.set_xy(10, 38)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(190, 7, '1. DESIGN INPUT DATA', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        # Material Properties Table
        pdf.set_font('Helvetica', '', 9)
        col_w = 47.5
        
        # Row 1
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(col_w, 6, f"Concrete: f'c = {fc_ksc:.0f} ksc", border=1, fill=True)
        pdf.cell(col_w, 6, f"f'c = {fc_mpa:.1f} MPa", border=1, fill=True)
        pdf.cell(col_w, 6, f"Steel Grade: {steel_grade}", border=1, fill=True)
        pdf.cell(col_w, 6, f"fy = {fy_mpa:.0f} MPa", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        # Row 2
        pdf.cell(col_w, 6, f"inline-size: b = {b_cm:.0f} cm", border=1, fill=True)
        pdf.cell(col_w, 6, f"Depth: h = {h_cm:.0f} cm", border=1, fill=True)
        pdf.cell(col_w, 6, f"Cover: c = {cover_cm:.1f} cm", border=1, fill=True)
        pdf.cell(col_w, 6, f"Ag = {Ag_cm2:.0f} cm2", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        # Row 3
        pdf.cell(col_w, 6, f"Main Bars: {total_bars}-{main_bar}", border=1, fill=True)
        pdf.cell(col_w, 6, f"Ast = {As_cm2:.2f} cm2", border=1, fill=True)
        pdf.cell(col_w, 6, f"Stirrup: {stirrup_bar}", border=1, fill=True)
        pdf.cell(col_w, 6, f"@ {stirrup_spacing_m*100:.0f} cm c/c", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        # Row 4 - Applied Loads
        pdf.set_fill_color(255, 255, 200)  # Light yellow for loads
        pdf.cell(col_w, 6, f"Pu = {Pu_ton:.2f} Ton", border=1, fill=True)
        pdf.cell(col_w, 6, f"Mu = {Mu_tonm:.2f} Ton-m", border=1, fill=True)
        if consider_slenderness:
            pdf.cell(col_w, 6, f"Lu = {Lu_m:.2f} m", border=1, fill=True)
            pdf.cell(col_w, 6, f"k = {k_factor:.2f}", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        else:
            pdf.cell(col_w * 2, 6, "Slenderness: Not Considered", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        # ----- CROSS-SECTION IMAGE -----
        y_after_table = pdf.get_y() + 3
        
        if fig_section is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig_section.savefig(tmp.name, dpi=150, bbox_inches='tight', 
                                   facecolor='white', edgecolor='none')
                temp_files.append(tmp.name)
                
                pdf.set_xy(120, y_after_table)
                pdf.set_font('Helvetica', 'B', 9)
                pdf.cell(80, 5, 'Cross-Section Detail:', new_x="LMARGIN", new_y="NEXT")
                pdf.image(tmp.name, x=115, y=pdf.get_y(), w=85)
        
        # ----- SECTION 2: DETAILED CALCULATIONS -----
        pdf.set_xy(10, y_after_table)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(100, 7, '2. SECTION PROPERTIES', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('Courier', '', 9)
        calc_x = 12
        
        # Gross Area Calculation
        pdf.set_xy(calc_x, pdf.get_y() + 1)
        pdf.cell(100, 5, f'Ag = b x h = {b_cm:.0f} x {h_cm:.0f} = {Ag_cm2:.0f} cm2', new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(100, 5, f'Ag = {Ag_mm2:.0f} mm2', new_x="LMARGIN", new_y="NEXT")
        
        # Steel Ratio Calculation
        pdf.set_x(calc_x)
        pdf.cell(100, 5, f'rho_g = Ast / Ag = {As_cm2:.2f} / {Ag_cm2:.0f}', new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(100, 5, f'rho_g = {rho_percent:.4f} = {rho_percent:.2f}%', new_x="LMARGIN", new_y="NEXT")
        
        # Check steel ratio limits
        pdf.set_x(calc_x)
        if 1.0 <= rho_percent <= 8.0:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(100, 5, f'Check: 1% <= {rho_percent:.2f}% <= 8% --> OK', new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(100, 5, f'Check: 1% <= {rho_percent:.2f}% <= 8% --> NG!', new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        
        # ----- SECTION 3: SLENDERNESS CHECK -----
        pdf.ln(3)
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(100, 7, '3. SLENDERNESS CHECK', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('Courier', '', 9)
        pdf.set_xy(calc_x, pdf.get_y() + 1)
        
        # Radius of gyration
        r_mm = 0.3 * h_mm
        r_cm = r_mm / 10
        pdf.cell(100, 5, f'r = 0.30 x h = 0.30 x {h_cm:.0f} = {r_cm:.1f} cm', new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(100, 5, f'r = {r_mm:.1f} mm', new_x="LMARGIN", new_y="NEXT")
        
        if consider_slenderness:
            Lu_mm = Lu_m * 1000
            kLu_r = (k_factor * Lu_mm) / r_mm
            
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'k x Lu / r = {k_factor:.2f} x {Lu_m*1000:.0f} / {r_mm:.1f}', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'k x Lu / r = {kLu_r:.1f}', new_x="LMARGIN", new_y="NEXT")
            
            # Slenderness limit check
            pdf.set_x(calc_x)
            if kLu_r > 22:
                pdf.set_text_color(255, 128, 0)  # Orange
                pdf.cell(100, 5, f'Check: {kLu_r:.1f} > 22 --> LONG COLUMN (Slender)', new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.set_text_color(0, 128, 0)
                pdf.cell(100, 5, f'Check: {kLu_r:.1f} <= 22 --> SHORT COLUMN', new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
        else:
            pdf.cell(100, 5, 'Slenderness effects not considered.', new_x="LMARGIN", new_y="NEXT")
        
        # ----- SECTION 4: MOMENT MAGNIFICATION (if applicable) -----
        pdf.ln(3)
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(100, 7, '4. MOMENT MAGNIFICATION (Non-Sway)', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('Courier', '', 9)
        pdf.set_xy(calc_x, pdf.get_y() + 1)
        
        if consider_slenderness and is_long_column:
            # Ec calculation
            Ec_mpa = 4700 * (fc_mpa ** 0.5)
            pdf.cell(100, 5, f'Ec = 4700 x sqrt(fc) = 4700 x sqrt({fc_mpa:.1f})', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Ec = {Ec_mpa:.0f} MPa', new_x="LMARGIN", new_y="NEXT")
            
            # Ig calculation
            Ig_mm4 = (b_mm * h_mm**3) / 12
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Ig = b x h^3 / 12 = {b_mm:.0f} x {h_mm:.0f}^3 / 12', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Ig = {Ig_mm4:.2e} mm4', new_x="LMARGIN", new_y="NEXT")
            
            # EI effective
            EI_eff = (0.4 * Ec_mpa * Ig_mm4) / (1 + beta_dns)
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'EI = 0.4 x Ec x Ig / (1 + beta_dns)', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'EI = 0.4 x {Ec_mpa:.0f} x {Ig_mm4:.2e} / (1 + {beta_dns})', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'EI = {EI_eff:.2e} N-mm2', new_x="LMARGIN", new_y="NEXT")
            
            # Critical buckling load
            kLu_mm = k_factor * Lu_m * 1000
            Pc_calc = (math.pi**2 * EI_eff) / (kLu_mm**2)
            Pc_ton_calc = Pc_calc / 9806.65
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Pc = pi^2 x EI / (k x Lu)^2', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Pc = {Pc_calc:.0f} N = {Pc_ton_calc:.1f} Ton', new_x="LMARGIN", new_y="NEXT")
            
            # Moment magnification factor
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'delta_ns = Cm / (1 - Pu / (0.75 x Pc))', new_x="LMARGIN", new_y="NEXT")
            Pu_N = Pu_ton * 9806.65
            denom = 1 - Pu_N / (0.75 * Pc_calc) if Pc_calc > 0 else 0
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'delta_ns = {Cm:.2f} / (1 - {Pu_ton:.1f} / (0.75 x {Pc_ton_calc:.1f}))', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.set_font('Courier', 'B', 10)
            pdf.cell(100, 5, f'delta_ns = {delta_ns:.3f}', new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Courier', '', 9)
            
            # Magnified moment
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Mc = delta_ns x Mu = {delta_ns:.3f} x {Mu_tonm:.2f}', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.set_font('Courier', 'B', 10)
            pdf.cell(100, 5, f'Mc = {Mc_tonm:.2f} Ton-m', new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(100, 5, 'Column is Short - No magnification required.', new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(calc_x)
            pdf.cell(100, 5, f'Mc = Mu = {Mu_tonm:.2f} Ton-m', new_x="LMARGIN", new_y="NEXT")
        
        # =====================================================================
        # PAGE 2: INTERACTION DIAGRAM & CAPACITY CHECK
        # =====================================================================
        pdf.add_page()
        
        # ----- HEADER (Page 2) -----
        pdf.set_fill_color(0, 51, 102)
        pdf.rect(0, 0, 210, 15, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_xy(10, 5)
        pdf.cell(0, 8, 'RC COLUMN DESIGN - P-M INTERACTION CHECK', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_text_color(0, 0, 0)
        
        # Project info bar
        pdf.set_xy(10, 18)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(63, 6, f'Column: {b_cm:.0f} x {h_cm:.0f} cm', border=1, fill=True)
        pdf.cell(63, 6, f'Rebar: {total_bars}-{main_bar}', border=1, fill=True)
        pdf.cell(64, 6, f'Page: 2 of 3', border=1, new_x="LMARGIN", new_y="NEXT", fill=True, align='R')
        
        # ----- P-M INTERACTION DIAGRAM -----
        pdf.set_xy(10, 27)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(190, 7, '5. P-M INTERACTION DIAGRAM', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        if fig_interaction is not None:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig_interaction.savefig(tmp.name, dpi=150, bbox_inches='tight',
                                       facecolor='white', edgecolor='none')
                temp_files.append(tmp.name)
                
                # Make diagram larger on page 2
                pdf.image(tmp.name, x=10, y=pdf.get_y() + 2, w=190, h=200)
        
        # ----- FOOTER NOTES (Page 2) -----
        pdf.set_xy(10, 275)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(190, 4, "P-M Interaction Diagram shows the column capacity envelope.", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_text_color(0, 0, 0)
        
        # =====================================================================
        # PAGE 3: CAPACITY CALCULATIONS & DESIGN CHECK
        # =====================================================================
        pdf.add_page()
        
        # ----- HEADER (Page 3) -----
        pdf.set_fill_color(0, 51, 102)
        pdf.rect(0, 0, 210, 15, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_xy(10, 5)
        pdf.cell(0, 8, 'RC COLUMN DESIGN - CAPACITY CHECK', new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_text_color(0, 0, 0)
        
        # Project info bar (Page 3)
        pdf.set_xy(10, 18)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(63, 6, f'Column: {b_cm:.0f} x {h_cm:.0f} cm', border=1, fill=True)
        pdf.cell(63, 6, f'Rebar: {total_bars}-{main_bar}', border=1, fill=True)
        pdf.cell(64, 6, f'Page: 3 of 3', border=1, new_x="LMARGIN", new_y="NEXT", fill=True, align='R')
        
        # ----- SECTION 6: KEY CAPACITY FORMULAS -----
        pdf.set_xy(10, 28)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(190, 7, '6. COLUMN CAPACITY (ACI 318)', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('Courier', '', 10)
        calc_x = 15
        line_h = 7
        pdf.set_xy(calc_x, pdf.get_y() + 2)
        
        # Max compression capacity formula
        pdf.cell(186, line_h, "Maximum Compression (Tied Column):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(186, line_h, "Pn,max = 0.80 x [0.85 x f'c x (Ag - Ast) + fy x Ast]", new_x="LMARGIN", new_y="NEXT")
        
        # Calculate P0
        P0_N = 0.85 * fc_mpa * (Ag_mm2 - As_mm2) + fy_mpa * As_mm2
        P0_ton = P0_N / 9806.65
        Pn_max_calc = 0.80 * P0_ton
        
        pdf.set_x(calc_x)
        pdf.cell(180, line_h, f"P0 = 0.85 x {fc_mpa:.1f} x ({Ag_mm2:.0f} - {As_mm2:.0f}) + {fy_mpa:.0f} x {As_mm2:.0f}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(180, line_h, f"P0 = {P0_N:.0f} N = {P0_ton:.1f} Ton", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.set_font('Courier', 'B', 10)
        pdf.cell(180, line_h, f"Pn,max = 0.80 x {P0_ton:.1f} = {Pn_max_calc:.1f} Ton", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font('Courier', '', 10)
        
        # Design capacity
        phi_comp = 0.65
        pdf.set_x(calc_x)
        pdf.cell(180, line_h, f"phi x Pn,max = {phi_comp} x {Pn_max_calc:.1f} = {phi_comp * Pn_max_calc:.1f} Ton", new_x="LMARGIN", new_y="NEXT")
        
        # ----- SECTION 7: DESIGN CHECK -----
        pdf.ln(8)
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(190, 7, '7. DESIGN CHECK', border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font('Courier', '', 10)
        pdf.set_xy(calc_x, pdf.get_y() + 3)
        
        # Applied loads
        pdf.cell(90, line_h, f"Applied Axial Load: Pu = {Pu_ton:.2f} Ton", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(90, line_h, f"Applied Moment: Mu = {Mu_tonm:.2f} Ton-m", new_x="LMARGIN", new_y="NEXT")
        
        if consider_slenderness and delta_ns > 1.0:
            pdf.set_x(calc_x)
            pdf.set_font('Courier', 'B', 10)
            pdf.cell(180, line_h, f"Magnified Moment: Mc = {Mc_tonm:.2f} Ton-m (delta_ns = {delta_ns:.3f})", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font('Courier', '', 10)
            design_moment = Mc_tonm
        else:
            design_moment = Mu_tonm
        
        # Safety verification
        pdf.ln(3)
        pdf.set_x(calc_x)
        pdf.cell(180, line_h, "Safety Verification:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(calc_x)
        pdf.cell(180, line_h, f"Design Point: (Pu, Mc) = ({Pu_ton:.2f} Ton, {design_moment:.2f} Ton-m)", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_x(calc_x)
        if is_safe:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(180, line_h, "Location: INSIDE the Design Capacity Curve", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(180, line_h, "Location: OUTSIDE the Design Capacity Curve", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        
        # D/C Ratio
        pdf.ln(3)
        pdf.set_x(calc_x)
        pdf.set_font('Courier', 'B', 11)
        pdf.cell(180, line_h, f"Demand/Capacity Ratio (D/C) = {dc_ratio:.3f} = {dc_ratio*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
        
        # ----- CONCLUSION BOX -----
        pdf.ln(10)
        pdf.set_x(10)
        
        if is_safe:
            pdf.set_fill_color(200, 255, 200)  # Light green
            pdf.set_draw_color(0, 128, 0)
            conclusion_text = "DESIGN STATUS: PASSED"
            status_detail = f"The column section is ADEQUATE for the applied loads."
        else:
            pdf.set_fill_color(255, 200, 200)  # Light red
            pdf.set_draw_color(200, 0, 0)
            conclusion_text = "DESIGN STATUS: FAILED"
            status_detail = f"The column section is INADEQUATE. Increase section or reinforcement."
        
        # Draw conclusion box
        box_y = pdf.get_y()
        pdf.rect(10, box_y, 190, 30, 'DF')
        
        pdf.set_xy(15, box_y + 5)
        pdf.set_font('Helvetica', 'B', 16)
        if is_safe:
            pdf.set_text_color(0, 100, 0)
        else:
            pdf.set_text_color(180, 0, 0)
        pdf.cell(180, 8, conclusion_text, new_x="LMARGIN", new_y="NEXT", align='C')
        
        pdf.set_xy(15, box_y + 15)
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(180, 6, status_detail, new_x="LMARGIN", new_y="NEXT", align='C')
        
        pdf.set_xy(15, box_y + 22)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(180, 5, f"D/C Ratio = {dc_ratio:.3f} ({dc_ratio*100:.1f}%) | Pu = {Pu_ton:.1f} T, Mc = {design_moment:.2f} T-m", new_x="LMARGIN", new_y="NEXT", align='C')
        
        # ----- FOOTER NOTES (Page 3) -----
        pdf.set_xy(10, 275)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(190, 4, "Notes: Design per ACI 318 / Thai Engineering Standards. phi = 0.65 (compression), 0.90 (tension).", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.cell(190, 4, "This calculation sheet is for preliminary design. Final design shall be verified by licensed engineer.", new_x="LMARGIN", new_y="NEXT", align='C')
        
        # Get PDF as bytes
        pdf_output = pdf.output()
        pdf_bytes = bytes(pdf_output) if isinstance(pdf_output, bytearray) else pdf_output
        
    finally:
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                os.unlink(tmp_file)
            except Exception:
                pass
    
    return pdf_bytes


# =============================================================================
# P-M INTERACTION DIAGRAM CALCULATIONS
# =============================================================================

@dataclass
class RebarLayer:
    """Represents a layer of reinforcement bars"""
    area: float      # Area of steel in this layer (mm²)
    distance: float  # Distance from extreme compression fiber (mm)


def calculate_beta1(fc_mpa: float) -> float:
    """
    Calculate β₁ factor for equivalent rectangular stress block.
    
    Per ACI 318:
    - β₁ = 0.85 for f'c ≤ 28 MPa (≈ 280 ksc)
    - β₁ reduces by 0.05 for each 7 MPa above 28 MPa
    - β₁ minimum = 0.65
    
    Note: 280 ksc ≈ 27.46 MPa, 70 ksc ≈ 6.87 MPa
    """
    if fc_mpa <= 28.0:  # ≈ 280 ksc
        return 0.85
    else:
        beta1 = 0.85 - 0.05 * (fc_mpa - 28.0) / 7.0
        return max(beta1, 0.65)


def calculate_phi_factor(epsilon_t: float, fy_mpa: float, Es: float = 200000.0) -> float:
    """
    Calculate strength reduction factor (φ) based on net tensile strain.
    
    Per ACI 318 for tied columns:
    - Compression controlled (εt ≤ εy): φ = 0.65
    - Transition zone (εy < εt < 0.005): Linear interpolation
    - Tension controlled (εt ≥ 0.005): φ = 0.90
    
    Args:
        epsilon_t: Net tensile strain in extreme tension steel
        fy_mpa: Steel yield strength in MPa
        Es: Modulus of elasticity of steel (default 200,000 MPa)
    
    Returns:
        Strength reduction factor φ
    """
    epsilon_y = fy_mpa / Es  # Yield strain
    
    # Compression controlled
    if epsilon_t <= epsilon_y:
        return 0.65
    
    # Tension controlled
    elif epsilon_t >= 0.005:
        return 0.90
    
    # Transition zone - linear interpolation
    else:
        phi = 0.65 + (0.90 - 0.65) * (epsilon_t - epsilon_y) / (0.005 - epsilon_y)
        return phi


def get_rebar_layers(
    b_mm: float,
    h_mm: float,
    cover_mm: float,
    stirrup_dia_mm: float,
    main_bar_dia_mm: float,
    n_bars_x: int,
    n_bars_y: int
) -> List[RebarLayer]:
    """
    Generate rebar layer positions for a rectangular column cross-section.
    
    Assumes bars are distributed around the perimeter with:
    - n_bars_x bars along width (b direction) on top and bottom
    - n_bars_y bars along depth (h direction) on left and right sides
    
    Returns:
        List of RebarLayer objects with area and distance from top fiber
    """
    # Calculate distances
    d_edge = cover_mm + stirrup_dia_mm + main_bar_dia_mm / 2  # Distance to first bar center
    
    single_bar_area = get_rebar_area(main_bar_dia_mm)
    
    layers = []
    
    # Create a dictionary to accumulate bars at each distance
    layer_dict = {}
    
    # Top layer (compression side) - n_bars_x bars
    dist = d_edge
    if dist not in layer_dict:
        layer_dict[dist] = 0
    layer_dict[dist] += n_bars_x * single_bar_area
    
    # Bottom layer (tension side) - n_bars_x bars
    dist = h_mm - d_edge
    if dist not in layer_dict:
        layer_dict[dist] = 0
    layer_dict[dist] += n_bars_x * single_bar_area
    
    # Intermediate layers (side bars) - only if n_bars_y > 2
    if n_bars_y > 2:
        n_intermediate = n_bars_y - 2
        # Spacing between intermediate bars
        spacing = (h_mm - 2 * d_edge) / (n_bars_y - 1)
        
        for i in range(1, n_bars_y - 1):
            dist = d_edge + i * spacing
            if dist not in layer_dict:
                layer_dict[dist] = 0
            # 2 bars (one on each side face)
            layer_dict[dist] += 2 * single_bar_area
    
    # Convert dictionary to list of RebarLayer objects, sorted by distance
    for dist in sorted(layer_dict.keys()):
        layers.append(RebarLayer(area=layer_dict[dist], distance=dist))
    
    return layers


def calculate_pm_point(
    c: float,
    fc_mpa: float,
    fy_mpa: float,
    b_mm: float,
    h_mm: float,
    beta1: float,
    rebar_layers: List[RebarLayer],
    Es: float = 200000.0,
    epsilon_cu: float = 0.003
) -> Tuple[float, float, float, float]:
    """
    Calculate Pn, Mn for a given neutral axis depth c.
    
    Uses Whitney stress block for concrete and elastic-perfectly plastic model for steel.
    
    Args:
        c: Neutral axis depth from compression fiber (mm)
        fc_mpa: Concrete compressive strength (MPa)
        fy_mpa: Steel yield strength (MPa)
        b_mm: Column width (mm)
        h_mm: Column depth (mm)
        beta1: Stress block factor
        rebar_layers: List of RebarLayer objects
        Es: Steel modulus of elasticity (MPa)
        epsilon_cu: Ultimate concrete strain (default 0.003)
    
    Returns:
        Tuple of (Pn, Mn, phi, epsilon_t)
    """
    # Handle edge cases
    if c <= 0:
        c = 0.001  # Small positive value
    
    # Depth of stress block
    a = beta1 * c
    
    # Limit stress block depth to section depth
    a_eff = min(a, h_mm)
    
    # Concrete compression force
    Cc = 0.85 * fc_mpa * a_eff * b_mm  # N
    
    # Concrete force acts at a/2 from compression fiber
    # Moment arm from centroid of section
    y_c = h_mm / 2 - a_eff / 2
    Mc = Cc * y_c  # N·mm
    
    # Steel forces and moments
    Ps_total = 0  # Total steel force (positive = compression)
    Ms_total = 0  # Total steel moment about centroid
    
    epsilon_t = 0  # Strain in extreme tension steel
    
    for layer in rebar_layers:
        d_i = layer.distance  # Distance from compression fiber
        As_i = layer.area
        
        # Steel strain (positive = compression, negative = tension)
        epsilon_s = epsilon_cu * (c - d_i) / c
        
        # Steel stress (limited by yield strength)
        if epsilon_s >= 0:  # Compression
            fs = min(epsilon_s * Es, fy_mpa)
        else:  # Tension
            fs = max(epsilon_s * Es, -fy_mpa)
        
        # Steel force (positive = compression)
        Fs = As_i * fs
        
        # For bars in compression zone, subtract displaced concrete area
        # (only if the stress block covers the bar)
        if d_i <= a_eff and epsilon_s >= 0:
            # Displaced concrete area contribution already counted in Cc
            # Adjust steel force: Fs_net = As * (fs - 0.85*f'c)
            Fs = As_i * (fs - 0.85 * fc_mpa)
        
        Ps_total += Fs
        
        # Moment about section centroid
        y_s = h_mm / 2 - d_i
        Ms_total += Fs * y_s
        
        # Track strain in extreme tension layer (largest d_i)
        if d_i == max(layer.distance for layer in rebar_layers):
            epsilon_t = -epsilon_s  # Convert to tension positive
    
    # Total nominal axial force (positive = compression)
    Pn = Cc + Ps_total
    
    # Total nominal moment (positive = causes compression on top)
    Mn = Mc + Ms_total
    
    # Calculate phi based on tensile strain
    phi = calculate_phi_factor(epsilon_t, fy_mpa, Es)
    
    return Pn, abs(Mn), phi, epsilon_t


def calculate_pure_compression(
    fc_mpa: float,
    fy_mpa: float,
    Ag_mm2: float,
    As_total_mm2: float
) -> Tuple[float, float]:
    """
    Calculate pure compression capacity (P0).
    
    P0 = 0.85 * f'c * (Ag - Ast) + fy * Ast
    
    For tied columns: φPn,max = 0.80 * φ * P0 (ACI 318 limit)
    """
    P0 = 0.85 * fc_mpa * (Ag_mm2 - As_total_mm2) + fy_mpa * As_total_mm2
    phi = 0.65  # Compression controlled
    
    # ACI 318 limits maximum axial strength to 80% for tied columns
    phi_Pn_max = 0.80 * phi * P0
    
    return P0, phi_Pn_max


def calculate_pure_tension(
    fy_mpa: float,
    As_total_mm2: float
) -> Tuple[float, float]:
    """
    Calculate pure tension capacity (Pnt).
    
    Pnt = fy * Ast (all steel yields in tension)
    """
    Pnt = fy_mpa * As_total_mm2
    phi = 0.90  # Tension controlled
    phi_Pnt = phi * Pnt
    
    return Pnt, phi_Pnt


def generate_pm_interaction_curve(
    fc_mpa: float,
    fy_mpa: float,
    b_mm: float,
    h_mm: float,
    cover_mm: float,
    stirrup_dia_mm: float,
    main_bar_dia_mm: float,
    n_bars_x: int,
    n_bars_y: int,
    num_points: int = 50
) -> dict:
    """
    Generate the complete P-M interaction diagram curve.
    
    Varies neutral axis depth from very large (pure compression) 
    to very small (pure tension) and calculates corresponding Pn, Mn values.
    
    Args:
        fc_mpa: Concrete compressive strength (MPa)
        fy_mpa: Steel yield strength (MPa)
        b_mm: Column width (mm)
        h_mm: Column depth (mm)
        cover_mm: Clear cover (mm)
        stirrup_dia_mm: Stirrup diameter (mm)
        main_bar_dia_mm: Main bar diameter (mm)
        n_bars_x: Number of bars along width
        n_bars_y: Number of bars along depth
        num_points: Number of points to generate for the curve
    
    Returns:
        Dictionary containing nominal and design (factored) values:
        - Pn, Mn: Nominal values
        - phi_Pn, phi_Mn: Design values (φ applied)
        - phi: Strength reduction factors
        - epsilon_t: Net tensile strains
        - c_values: Neutral axis depths used
    """
    # Calculate parameters
    beta1 = calculate_beta1(fc_mpa)
    
    # Get rebar layers
    rebar_layers = get_rebar_layers(
        b_mm, h_mm, cover_mm, stirrup_dia_mm, main_bar_dia_mm, n_bars_x, n_bars_y
    )
    
    # Calculate total steel area
    As_total = sum(layer.area for layer in rebar_layers)
    Ag = b_mm * h_mm
    
    # Effective depth to extreme tension steel
    d = max(layer.distance for layer in rebar_layers)
    
    # Calculate pure compression point
    P0, phi_Pn_max = calculate_pure_compression(fc_mpa, fy_mpa, Ag, As_total)
    
    # Calculate pure tension point
    Pnt, phi_Pnt = calculate_pure_tension(fy_mpa, As_total)
    
    # Initialize result lists
    results = {
        'Pn': [],
        'Mn': [],
        'phi_Pn': [],
        'phi_Mn': [],
        'phi': [],
        'epsilon_t': [],
        'c_values': [],
        'P0': P0,
        'phi_Pn_max': phi_Pn_max,
        'Pnt': Pnt,
        'phi_Pnt': phi_Pnt,
        'beta1': beta1,
        'As_total': As_total,
        'rebar_layers': rebar_layers
    }
    
    # Generate c values from large (compression) to small (tension)
    # Use logarithmic spacing for better distribution near transition zones
    c_max = 5 * h_mm  # Very large c for pure compression
    c_min = 0.01 * h_mm  # Very small c for tension
    
    # Create array of c values
    c_values = np.concatenate([
        np.linspace(c_max, h_mm, num_points // 4),
        np.linspace(h_mm, d, num_points // 4),
        np.linspace(d, 0.5 * d, num_points // 4),
        np.linspace(0.5 * d, c_min, num_points // 4)
    ])
    
    # Remove duplicates and sort
    c_values = np.unique(c_values)[::-1]
    
    for c in c_values:
        Pn, Mn, phi, epsilon_t = calculate_pm_point(
            c, fc_mpa, fy_mpa, b_mm, h_mm, beta1, rebar_layers
        )
        
        # Apply phi factor
        phi_Pn = phi * Pn
        phi_Mn = phi * Mn
        
        # Apply maximum compression limit for tied columns
        if phi_Pn > phi_Pn_max:
            phi_Pn = phi_Pn_max
        
        results['Pn'].append(Pn)
        results['Mn'].append(Mn)
        results['phi_Pn'].append(phi_Pn)
        results['phi_Mn'].append(phi_Mn)
        results['phi'].append(phi)
        results['epsilon_t'].append(epsilon_t)
        results['c_values'].append(c)
    
    # Add pure tension point
    results['Pn'].append(-Pnt)
    results['Mn'].append(0)
    results['phi_Pn'].append(-phi_Pnt)
    results['phi_Mn'].append(0)
    results['phi'].append(0.90)
    results['epsilon_t'].append(fy_mpa / 200000 + 0.01)  # Large tension strain
    results['c_values'].append(0)
    
    # Convert to numpy arrays
    for key in ['Pn', 'Mn', 'phi_Pn', 'phi_Mn', 'phi', 'epsilon_t', 'c_values']:
        results[key] = np.array(results[key])
    
    return results


def find_balanced_point(results: dict) -> dict:
    """
    Find the balanced condition point where φ transitions from 0.65 to higher values.
    
    The balanced point occurs when εt = εy (yield strain of steel).
    For practical purposes, this is approximately where φ starts increasing from 0.65.
    """
    phi_values = results['phi']
    
    # Find index where phi just starts to increase from 0.65
    for i in range(len(phi_values) - 1):
        if phi_values[i] <= 0.65 < phi_values[i + 1]:
            return {
                'index': i,
                'Pn': results['Pn'][i],
                'Mn': results['Mn'][i],
                'phi_Pn': results['phi_Pn'][i],
                'phi_Mn': results['phi_Mn'][i],
                'c': results['c_values'][i]
            }
    
    # If not found, return the point closest to phi = 0.65
    idx = np.argmin(np.abs(phi_values - 0.65))
    return {
        'index': idx,
        'Pn': results['Pn'][idx],
        'Mn': results['Mn'][idx],
        'phi_Pn': results['phi_Pn'][idx],
        'phi_Mn': results['phi_Mn'][idx],
        'c': results['c_values'][idx]
    }


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="RC Column Design",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Rectangular Reinforced Concrete Column Design")
    st.markdown("**Based on Thai Engineering Standards (ACI 318 Metric)**")
    st.markdown("**Developed by: A.THONGCHART**")
    
    # Legal Disclaimer / ข้อเสนอแนะตามหลักกฎหมาย
    with st.expander("⚠️ ข้อจำกัดความรับผิดชอบ / Disclaimer", expanded=False):
        st.warning("""
        **ข้อจำกัดความรับผิดชอบ (Disclaimer)**
        
        🔸 โปรแกรมนี้เป็นเครื่องมือช่วยในการคำนวณเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการตัดสินใจทางวิศวกรรมโดยผู้เชี่ยวชาญได้
        
        🔸 ผู้ใช้งานต้องมีพื้นฐานความรู้ด้านวิศวกรรมโครงสร้างและการออกแบบคอนกรีตเสริมเหล็กตามมาตรฐาน ACI 318
        
        🔸 การคำนวณและการออกแบบทั้งหมดเป็นความรับผิดชอบของผู้ใช้งานแต่เพียงผู้เดียว
        
        🔸 ผู้พัฒนาไม่รับผิดชอบต่อความเสียหายใดๆ ที่เกิดจากการใช้งานโปรแกรมนี้
        
        🔸 ผลการคำนวณควรได้รับการตรวจสอบโดยวิศวกรโยธาที่มีใบอนุญาตประกอบวิชาชีพ (กว.) ก่อนนำไปใช้งานจริง
        
        ---
        
        **Legal Disclaimer**
        
        🔸 This program is intended as a preliminary calculation tool only and cannot replace engineering judgment by qualified professionals.
        
        🔸 Users must have fundamental knowledge in structural engineering and reinforced concrete design according to ACI 318 standards.
        
        🔸 All calculations and designs are the sole responsibility of the user.
        
        🔸 The developer is not liable for any damages arising from the use of this program.
        
        🔸 Calculation results should be verified by a licensed civil engineer before actual implementation.
        """)
    
    st.markdown("---")
    
    # =========================================================================
    # SIDEBAR - INPUT SECTION
    # =========================================================================
    
    st.sidebar.header("📐 Design Inputs")
    
    # -------------------------------------------------------------------------
    # Material Properties
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Material Properties")
    
    # Concrete Strength
    fc_ksc = st.sidebar.number_input(
        "Concrete Strength, $f'_c$ (ksc)",
        min_value=100.0,
        max_value=600.0,
        value=240.0,
        step=10.0,
        help="Common values: 180, 210, 240, 280, 320, 350 ksc"
    )
    
    # Steel Grade Selection
    steel_grade = st.sidebar.selectbox(
        "Steel Grade, $f_y$",
        options=list(STEEL_GRADES.keys()),
        format_func=lambda x: STEEL_GRADES[x]["description"],
        index=1,  # Default to SD40
        help="Select reinforcement steel grade"
    )
    fy_mpa = STEEL_GRADES[steel_grade]["fy_mpa"]
    
    # -------------------------------------------------------------------------
    # Column Dimensions
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Column Dimensions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        b_cm = st.number_input(
            "Width, $b$ (cm)",
            min_value=15.0,
            max_value=200.0,
            value=30.0,
            step=5.0,
            help="Column width in cm"
        )
    
    with col2:
        h_cm = st.number_input(
            "Depth, $h$ (cm)",
            min_value=15.0,
            max_value=200.0,
            value=40.0,
            step=5.0,
            help="Column depth in cm"
        )
    
    # Clear Cover
    cover_cm = st.sidebar.number_input(
        "Clear Cover (cm)",
        min_value=2.0,
        max_value=10.0,
        value=4.0,
        step=0.5,
        help="Clear cover to stirrups (typical: 4 cm)"
    )
    
    # -------------------------------------------------------------------------
    # Reinforcement Configuration
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Reinforcement Configuration")
    
    # Main Bar Diameter
    main_bar = st.sidebar.selectbox(
        "Main Bar Diameter",
        options=list(REBAR_DIAMETERS.keys()),
        index=2,  # Default to DB20
        help="Diameter of main longitudinal bars"
    )
    main_bar_dia_mm = REBAR_DIAMETERS[main_bar]
    
    # Stirrup Diameter (Round Bars - RB/SR24)
    stirrup_bar = st.sidebar.selectbox(
        "Stirrup Bar (RB - SR24)",
        options=list(STIRRUP_BARS.keys()),
        format_func=lambda x: STIRRUP_BARS[x]["description"],
        index=1,  # Default to RB9
        help="Round bar stirrups - SR24 grade (fy = 235 MPa ≈ 2400 ksc)"
    )
    stirrup_dia_mm = STIRRUP_BARS[stirrup_bar]["diameter"]
    fy_stirrup_mpa = STIRRUP_BARS[stirrup_bar]["fy_mpa"]  # Always 235 MPa for RB
    
    # Stirrup Spacing
    stirrup_spacing_cm = st.sidebar.number_input(
        "Stirrup Spacing (cm)",
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=2.5,
        help="Spacing of stirrups along column length"
    )
    stirrup_spacing_m = stirrup_spacing_cm / 100  # Convert to meters
    
    # Number of Bars
    st.sidebar.markdown("**Number of Bars**")
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        n_bars_x = st.number_input(
            "Bars in X-dir",
            min_value=2,
            max_value=20,
            value=3,
            step=1,
            help="Number of bars along width (b)"
        )
    
    with col4:
        n_bars_y = st.number_input(
            "Bars in Y-dir",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            help="Number of bars along depth (h)"
        )
    
    # Calculate total number of bars (perimeter arrangement)
    # Corner bars counted once: 2*n_x + 2*n_y - 4
    total_bars = 2 * n_bars_x + 2 * (n_bars_y - 2)
    if n_bars_y <= 2:
        total_bars = 2 * n_bars_x
    
    # -------------------------------------------------------------------------
    # Applied Loads (Design Loads)
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Applied Loads (Factored)")
    
    st.sidebar.markdown("**Design Loads ($P_u$, $M_u$)**")
    
    Pu_ton = st.sidebar.number_input(
        "Axial Load, $P_u$ (Ton)",
        min_value=-500.0,
        max_value=2000.0,
        value=50.0,
        step=5.0,
        help="Factored axial load (positive = compression, negative = tension)"
    )
    
    Mu_tonm = st.sidebar.number_input(
        "Moment, $M_u$ (Ton·m)",
        min_value=0.0,
        max_value=500.0,
        value=5.0,
        step=1.0,
        help="Factored bending moment"
    )
    
    # Convert applied loads to SI units
    Pu_N = Pu_ton * TON_TO_N
    Mu_Nmm = Mu_tonm * TON_TO_N * 1000  # Ton·m to N·mm
    
    # -------------------------------------------------------------------------
    # Slenderness Effects (Long Column)
    # -------------------------------------------------------------------------
    st.sidebar.subheader("Slenderness Effects")
    
    consider_slenderness = st.sidebar.checkbox(
        "Consider Slenderness Effect?",
        value=False,
        help="Enable moment magnification for slender (long) columns"
    )
    
    Lu_m = st.sidebar.number_input(
        "Unsupported Height, $L_u$ (m)",
        min_value=0.5,
        max_value=20.0,
        value=3.0,
        step=0.1,
        help="Clear height between lateral supports"
    )
    
    k_factor = st.sidebar.number_input(
        "Effective Length Factor, $k$",
        min_value=0.5,
        max_value=2.5,
        value=1.0,
        step=0.1,
        help="k=1.0 for pinned-pinned, k=0.7 for fixed-pinned, k=0.5 for fixed-fixed, k=2.0 for cantilever"
    )
    
    # Advanced slenderness parameters (in expander)
    with st.sidebar.expander("Advanced Slenderness Parameters"):
        beta_dns = st.number_input(
            "Sustained Load Factor, $β_{dns}$",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Ratio of sustained load to total load (typically 0.6)"
        )
        
        Cm = st.number_input(
            "Moment Factor, $C_m$",
            min_value=0.4,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Equivalent uniform moment factor (1.0 is conservative)"
        )
    
    # Convert Lu to mm
    Lu_mm = Lu_m * 1000
    
    # =========================================================================
    # UNIT CONVERSIONS (to N and mm)
    # =========================================================================
    
    # Convert all inputs to SI units (N, mm, MPa)
    fc_mpa = convert_fc_ksc_to_mpa(fc_ksc)
    b_mm = convert_dimension_cm_to_mm(b_cm)
    h_mm = convert_dimension_cm_to_mm(h_cm)
    cover_mm = convert_dimension_cm_to_mm(cover_cm)
    
    # Calculate effective depth
    d_mm = h_mm - cover_mm - stirrup_dia_mm - main_bar_dia_mm / 2
    d_prime_mm = cover_mm + stirrup_dia_mm + main_bar_dia_mm / 2
    
    # Calculate steel areas
    As_total_mm2 = calculate_total_steel_area(total_bars, main_bar_dia_mm)
    Ag_mm2 = b_mm * h_mm  # Gross area
    
    # Steel ratio
    rho = As_total_mm2 / Ag_mm2
    
    # =========================================================================
    # SLENDERNESS CALCULATIONS
    # =========================================================================
    
    # Calculate slenderness parameters
    Ec_mpa = calculate_Ec(fc_mpa)
    r_mm = 0.3 * h_mm  # Radius of gyration for rectangular section
    slenderness_ratio = calculate_slenderness_ratio(k_factor, Lu_mm, h_mm)
    is_slender = check_slenderness_limit(slenderness_ratio, limit=22.0)
    
    # Calculate critical buckling load and magnification factor
    Pc_N = calculate_critical_buckling_load(Ec_mpa, b_mm, h_mm, k_factor, Lu_mm, beta_dns)
    
    # Magnification factor (only meaningful for compression)
    if Pu_N > 0 and consider_slenderness:
        delta_ns = calculate_moment_magnification_factor(Pu_N, Pc_N, Cm)
    else:
        delta_ns = 1.0
    
    # Magnified moment
    Mc_tonm = delta_ns * Mu_tonm  # Magnified moment in Ton·m
    Mc_Nmm = delta_ns * Mu_Nmm    # Magnified moment in N·mm
    
    # =========================================================================
    # MAIN CONTENT - DISPLAY RESULTS
    # =========================================================================
    
    # Display Input Summary
    col_left, col_right = st.columns(2)
    
    col_left.subheader("📋 Input Summary")
    
    col_left.markdown("**Material Properties:**")
    col_left.write(f"- Concrete Strength: $f'_c$ = {fc_ksc:.0f} ksc = **{fc_mpa:.2f} MPa**")
    col_left.write(f"- Steel Grade: {STEEL_GRADES[steel_grade]['description']}")
    col_left.write(f"- Steel Yield Strength: $f_y$ = **{fy_mpa:.0f} MPa**")
    
    col_left.markdown("**Column Dimensions:**")
    col_left.write(f"- inline-size: $b$ = {b_cm:.0f} cm = **{b_mm:.0f} mm**")
    col_left.write(f"- Depth: $h$ = {h_cm:.0f} cm = **{h_mm:.0f} mm**")
    col_left.write(f"- Clear Cover = {cover_cm:.1f} cm = **{cover_mm:.0f} mm**")
    
    col_left.markdown("**Reinforcement:**")
    col_left.write(f"- Main Bars: {total_bars}-{main_bar} ({n_bars_x} bars × {n_bars_y} bars)")
    col_left.write(f"- Main Bar Diameter: **{main_bar_dia_mm} mm**")
    col_left.write(f"- Stirrup: **{stirrup_bar} @ {stirrup_spacing_cm:.0f} cm** (dia = {stirrup_dia_mm} mm, $f_{{yv}}$ = {fy_stirrup_mpa} MPa)")
    
    col_right.subheader("📊 Calculated Values")
    
    col_right.markdown("**Cross-Section Properties:**")
    col_right.write(f"- Gross Area: $A_g$ = {Ag_mm2:,.0f} mm² = **{Ag_mm2/100:,.0f} cm²**")
    col_right.write(f"- Total Steel Area: $A_s$ = **{As_total_mm2:,.0f} mm²**")
    col_right.write(f"- Steel Ratio: $\\rho$ = **{rho*100:.2f}%**")
    
    col_right.markdown("**Effective Depths:**")
    col_right.write(f"- Effective Depth: $d$ = **{d_mm:.1f} mm**")
    col_right.write(f"- Compression Steel Depth: $d'$ = **{d_prime_mm:.1f} mm**")
    
    # Check steel ratio limits (ACI 318)
    rho_min = 0.01  # 1% minimum
    rho_max = 0.08  # 8% maximum (can be 6% at lap splices)
    
    col_right.markdown("**Code Compliance (ACI 318):**")
    if rho < rho_min:
        col_right.error(f"⚠️ Steel ratio {rho*100:.2f}% < {rho_min*100:.0f}% (minimum)")
    elif rho > rho_max:
        col_right.error(f"⚠️ Steel ratio {rho*100:.2f}% > {rho_max*100:.0f}% (maximum)")
    else:
        col_right.success(f"✅ Steel ratio {rho*100:.2f}% is within limits ({rho_min*100:.0f}% - {rho_max*100:.0f}%)")
    
    # =========================================================================
    # COLUMN CROSS-SECTION VISUALIZATION
    # =========================================================================
    
    st.markdown("---")
    st.subheader("🏗️ Column Cross-Section")
    
    # Create columns for cross-section plot and additional info
    col_section, col_info = st.columns([2, 1])
    
    with col_section:
        # Generate cross-section figure (stirrup_spacing_m is from sidebar input)
        fig_section = plot_column_section(
            b_mm=b_mm,
            h_mm=h_mm,
            cover_mm=cover_mm,
            main_bar_dia_mm=main_bar_dia_mm,
            stirrup_dia_mm=stirrup_dia_mm,
            n_bars_x=n_bars_x,
            n_bars_y=n_bars_y,
            stirrup_spacing_m=stirrup_spacing_m,
            stirrup_bar_name=stirrup_bar,
            main_bar_name=main_bar
        )
        
        st.pyplot(fig_section)
    
    with col_info:
        st.markdown("**Section Details:**")
        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | Width (b) | {b_cm:.0f} cm |
        | Depth (h) | {h_cm:.0f} cm |
        | Cover | {cover_cm:.1f} cm |
        | Main Bars | {total_bars}-{main_bar} |
        | Stirrups | {stirrup_bar} @ {stirrup_spacing_m*100:.0f} cm |
        | Ag | {Ag_mm2/100:,.0f} cm² |
        | As | {As_total_mm2:,.0f} mm² |
        | ρ | {rho*100:.2f}% |
        """)
        
        st.markdown("**Bar Layout:**")
        st.write(f"- X-direction: {n_bars_x} bars")
        st.write(f"- Y-direction: {n_bars_y} bars")
        st.write(f"- Total: {total_bars} bars")
        
        # Bar spacing calculation
        bar_center_offset = cover_mm + stirrup_dia_mm + main_bar_dia_mm / 2
        if n_bars_x > 1:
            spacing_x = (b_mm - 2 * bar_center_offset) / (n_bars_x - 1)
            st.write(f"- Spacing (X): {spacing_x:.1f} mm = {spacing_x/10:.1f} cm")
        if n_bars_y > 1:
            spacing_y = (h_mm - 2 * bar_center_offset) / (n_bars_y - 1)
            st.write(f"- Spacing (Y): {spacing_y:.1f} mm = {spacing_y/10:.1f} cm")
    
    # =========================================================================
    # SLENDERNESS EFFECTS DISPLAY
    # =========================================================================
    
    if consider_slenderness:
        st.markdown("---")
        st.subheader("📏 Slenderness Effects (Long Column Analysis)")
        
        col_slender1, col_slender2 = st.columns(2)
        
        with col_slender1:
            st.markdown("**Slenderness Parameters:**")
            st.write(f"- Unsupported block-size: $L_u$ = {Lu_m:.2f} m = {Lu_mm:.0f} mm")
            st.write(f"- Effective Length Factor: $k$ = {k_factor:.2f}")
            st.write(f"- Effective Length: $kL_u$ = {k_factor * Lu_mm:.0f} mm = {k_factor * Lu_m:.2f} m")
            st.write(f"- Radius of Gyration: $r = 0.3h$ = {r_mm:.1f} mm")
            st.write(f"- **Slenderness Ratio: $kL_u/r$ = {slenderness_ratio:.1f}**")
        
        with col_slender2:
            st.markdown("**Column Classification:**")
            if is_slender:
                st.warning(f"⚠️ **SLENDER COLUMN** ($kL_u/r$ = {slenderness_ratio:.1f} > 22)")
                st.write("Moment magnification is required.")
            else:
                st.success(f"✅ **SHORT COLUMN** ($kL_u/r$ = {slenderness_ratio:.1f} ≤ 22)")
                st.write("Slenderness effects can be neglected.")
            
            st.markdown("**Buckling Analysis:**")
            st.write(f"- Modulus of Elasticity: $E_c$ = {Ec_mpa:.0f} MPa")
            st.write(f"- Sustained Load Factor: $β_{{dns}}$ = {beta_dns:.2f}")
            st.write(f"- Critical Buckling Load: $P_c$ = {Pc_N/1000:.0f} kN = {Pc_N/TON_TO_N:.1f} Ton")
        
        # Show magnification results
        st.markdown("**Moment Magnification:**")
        col_mag1, col_mag2, col_mag3 = st.columns(3)
        
        with col_mag1:
            st.metric("Magnification Factor", f"δns = {delta_ns:.3f}")
        
        with col_mag2:
            st.metric("Original Moment", f"Mu = {Mu_tonm:.2f} Ton·m")
        
        with col_mag3:
            if delta_ns > 1.0:
                st.metric("Magnified Moment", f"Mc = {Mc_tonm:.2f} Ton·m", 
                         delta=f"+{(delta_ns-1)*100:.1f}%", delta_color="inverse")
            else:
                st.metric("Magnified Moment", f"Mc = {Mc_tonm:.2f} Ton·m")
        
        # Warning for unstable column
        if delta_ns == float('inf'):
            st.error("❌ **COLUMN IS UNSTABLE!** $P_u > 0.75 P_c$ - Column will buckle. Increase section size or reduce load.")
        elif delta_ns > 2.0:
            st.warning(f"⚠️ High magnification factor (δns = {delta_ns:.2f}). Consider increasing column stiffness.")
    
    # =========================================================================
    # P-M INTERACTION DIAGRAM
    # =========================================================================
    
    st.markdown("---")
    st.subheader("📈 P-M Interaction Diagram (Uniaxial Bending)")
    
    # Generate interaction curve
    pm_results = generate_pm_interaction_curve(
        fc_mpa=fc_mpa,
        fy_mpa=fy_mpa,
        b_mm=b_mm,
        h_mm=h_mm,
        cover_mm=cover_mm,
        stirrup_dia_mm=stirrup_dia_mm,
        main_bar_dia_mm=main_bar_dia_mm,
        n_bars_x=n_bars_x,
        n_bars_y=n_bars_y,
        num_points=60
    )
    
    # Convert units for display
    # Force: N to kN and Ton
    # Moment: N·mm to kN·m and Ton·m
    
    phi_Pn_kN = pm_results['phi_Pn'] / 1000  # N to kN
    phi_Mn_kNm = pm_results['phi_Mn'] / 1e6   # N·mm to kN·m
    
    phi_Pn_ton = pm_results['phi_Pn'] / TON_TO_N  # N to Ton
    phi_Mn_tonm = pm_results['phi_Mn'] / (TON_TO_N * 1000)  # N·mm to Ton·m
    
    Pn_kN = pm_results['Pn'] / 1000
    Mn_kNm = pm_results['Mn'] / 1e6
    
    # Find key points
    balanced_point = find_balanced_point(pm_results)
    
    # Maximum moment point
    max_Mn_idx = np.argmax(pm_results['phi_Mn'])
    
    # Display key values
    col_pm1, col_pm2 = st.columns(2)
    
    with col_pm1:
        st.markdown("**Key Capacity Values (Design - φ applied):**")
        st.write(f"- Max Compression: $\\phi P_{{n,max}}$ = **{pm_results['phi_Pn_max']/1000:.0f} kN** = {pm_results['phi_Pn_max']/TON_TO_N:.1f} Ton")
        st.write(f"- Pure Compression: $P_0$ = {pm_results['P0']/1000:.0f} kN = {pm_results['P0']/TON_TO_N:.1f} Ton")
        st.write(f"- Max Tension: $\\phi P_{{nt}}$ = **{pm_results['phi_Pnt']/1000:.0f} kN** = {pm_results['phi_Pnt']/TON_TO_N:.1f} Ton")
    
    with col_pm2:
        st.markdown("**Material Parameters:**")
        st.write(f"- β₁ = **{pm_results['beta1']:.3f}**")
        st.write(f"- Total Steel Area: $A_s$ = {pm_results['As_total']:.0f} mm²")
        st.write(f"- Number of rebar layers: {len(pm_results['rebar_layers'])}")
    
    # Balanced point info
    st.markdown("**Balanced Condition:**")
    st.write(f"- $\\phi P_b$ = {balanced_point['phi_Pn']/1000:.0f} kN ({balanced_point['phi_Pn']/TON_TO_N:.1f} Ton), "
             f"$\\phi M_b$ = {balanced_point['phi_Mn']/1e6:.1f} kN·m ({balanced_point['phi_Mn']/(TON_TO_N*1000):.2f} Ton·m)")
    st.write(f"- Neutral axis depth at balanced: c = {balanced_point['c']:.1f} mm")
    
    # Maximum moment point
    st.markdown("**Maximum Moment:**")
    st.write(f"- $\\phi M_{{max}}$ = **{phi_Mn_kNm[max_Mn_idx]:.1f} kN·m** ({phi_Mn_tonm[max_Mn_idx]:.2f} Ton·m) "
             f"at $\\phi P_n$ = {phi_Pn_kN[max_Mn_idx]:.0f} kN ({phi_Pn_ton[max_Mn_idx]:.1f} Ton)")
    
    # =========================================================================
    # CHECK APPLIED LOAD vs CAPACITY
    # =========================================================================
    
    st.markdown("---")
    st.subheader("🔍 Design Check: Applied Load vs Capacity")
    
    # Function to check if point is inside the interaction curve
    def check_load_inside_curve(Pu_ton, Mu_tonm, phi_Pn_ton, phi_Mn_tonm):
        """
        Check if the applied load point (Pu, Mu) is inside the interaction curve.
        Uses ray casting algorithm.
        """
        from matplotlib.path import Path
        
        # Create path from interaction curve points
        # Need to close the curve by adding back to start
        vertices = list(zip(phi_Mn_tonm, phi_Pn_ton))
        vertices.append(vertices[0])  # Close the path
        
        path = Path(vertices)
        point = (Mu_tonm, Pu_ton)
        
        return path.contains_point(point)
    
    # Check if load is inside capacity curve
    is_safe = check_load_inside_curve(Pu_ton, Mu_tonm, phi_Pn_ton, phi_Mn_tonm)
    
    # Calculate capacity ratio (approximate)
    # Find the intersection point on the curve at the same angle
    def calculate_capacity_ratio(Pu_ton, Mu_tonm, phi_Pn_ton, phi_Mn_tonm):
        """Calculate approximate demand/capacity ratio using radial line method."""
        if Mu_tonm == 0 and Pu_ton == 0:
            return 0.0
        
        # For points along the curve, find the one with similar P/M ratio
        angle_applied = np.arctan2(Pu_ton, Mu_tonm) if Mu_tonm != 0 else np.pi/2 if Pu_ton > 0 else -np.pi/2
        
        # Calculate angles for all curve points
        angles_curve = np.arctan2(phi_Pn_ton, phi_Mn_tonm)
        
        # Find the closest angle on the curve
        idx = np.argmin(np.abs(angles_curve - angle_applied))
        
        # Calculate distance from origin to applied load
        dist_applied = np.sqrt(Pu_ton**2 + Mu_tonm**2)
        
        # Calculate distance from origin to curve point
        dist_capacity = np.sqrt(phi_Pn_ton[idx]**2 + phi_Mn_tonm[idx]**2)
        
        if dist_capacity == 0:
            return float('inf')
        
        return dist_applied / dist_capacity
    
    # Use magnified moment for safety check when slenderness is considered
    if consider_slenderness and delta_ns != float('inf'):
        M_design_tonm = Mc_tonm  # Use magnified moment
    else:
        M_design_tonm = Mu_tonm  # Use original moment
    
    # Check if load is inside capacity curve (using design moment)
    is_safe = check_load_inside_curve(Pu_ton, M_design_tonm, phi_Pn_ton, phi_Mn_tonm)
    
    # Calculate capacity ratio using design moment
    capacity_ratio = calculate_capacity_ratio(Pu_ton, M_design_tonm, phi_Pn_ton, phi_Mn_tonm)
    
    col_check1, col_check2 = st.columns(2)
    
    with col_check1:
        st.markdown("**Applied Loads:**")
        st.write(f"- $P_u$ = **{Pu_ton:.1f} Ton** ({Pu_ton * TON_TO_N / 1000:.1f} kN)")
        st.write(f"- $M_u$ = **{Mu_tonm:.2f} Ton·m** ({Mu_tonm * TON_TO_N / 1e3:.1f} kN·m)")
        
        if consider_slenderness and delta_ns > 1.0:
            st.markdown("**Magnified Design Moment:**")
            st.write(f"- $δ_{{ns}}$ = **{delta_ns:.3f}**")
            st.write(f"- $M_c = δ_{{ns}} × M_u$ = **{Mc_tonm:.2f} Ton·m** ({Mc_tonm * TON_TO_N / 1e3:.1f} kN·m)")
            st.info("📐 Using magnified moment $M_c$ for design check")
    
    with col_check2:
        st.markdown("**Design Status:**")
        
        # Check for unstable column first
        if consider_slenderness and delta_ns == float('inf'):
            st.error("❌ **UNSTABLE** - Column will buckle ($P_u > 0.75P_c$)")
            st.write("Increase section size or reduce axial load!")
        elif is_safe:
            st.success(f"✅ **SAFE** - Load point is INSIDE the interaction curve")
            st.write(f"- Capacity Ratio (D/C): **{capacity_ratio:.2%}**")
        else:
            st.error(f"❌ **UNSAFE** - Load point is OUTSIDE the interaction curve")
            st.write(f"- Capacity Ratio (D/C): **{capacity_ratio:.2%}** > 100%")
        
        if capacity_ratio <= 0.7 and is_safe:
            st.info("💡 Section may be over-designed. Consider reducing size or reinforcement.")
        elif capacity_ratio > 0.9 and is_safe:
            st.warning("⚠️ Close to capacity limit. Consider increasing safety margin.")
    
    # =========================================================================
    # PLOT INTERACTION DIAGRAM
    # =========================================================================
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches
    
    # Nominal values in Ton units
    Pn_ton = pm_results['Pn'] / TON_TO_N
    Mn_tonm = pm_results['Mn'] / (TON_TO_N * 1000)
    
    # Create figure with single main plot (Thai Units - Ton, Ton·m)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Set Thai-English labels style
    plt.rcParams['font.size'] = 11
    
    # Plot Nominal Capacity curve (dashed)
    ax.plot(Mn_tonm, Pn_ton, 'b--', linewidth=1.5, alpha=0.6, 
            label='Nominal Capacity (Pn, Mn)')
    
    # Plot Design Capacity curve (solid with phi)
    ax.plot(phi_Mn_tonm, phi_Pn_ton, 'b-', linewidth=2.5, 
            label='Design Capacity (φPn, φMn)')
    
    # Fill the safe zone
    ax.fill_between(phi_Mn_tonm, phi_Pn_ton, alpha=0.15, color='blue')
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    
    # Max compression line
    phi_Pn_max_ton = pm_results['phi_Pn_max'] / TON_TO_N
    ax.axhline(y=phi_Pn_max_ton, color='red', linestyle=':', 
               linewidth=1.5, alpha=0.8)
    ax.annotate(f'φPn,max = {phi_Pn_max_ton:.1f} Ton', 
                xy=(0.02, phi_Pn_max_ton), 
                xycoords=('axes fraction', 'data'),
                fontsize=9, color='red', va='bottom')
    
    # Balanced point marker
    balanced_Mn_tonm = balanced_point['phi_Mn'] / (TON_TO_N * 1000)
    balanced_Pn_ton = balanced_point['phi_Pn'] / TON_TO_N
    ax.plot(balanced_Mn_tonm, balanced_Pn_ton, 'ro', markersize=10, 
            markeredgecolor='darkred', markeredgewidth=2, label='Balanced Point')
    ax.annotate(f'Balanced\n({balanced_Mn_tonm:.1f}, {balanced_Pn_ton:.0f})', 
                xy=(balanced_Mn_tonm, balanced_Pn_ton),
                xytext=(15, -25), textcoords='offset points',
                fontsize=9, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1))
    
    # Max moment point marker
    max_Mn_tonm = phi_Mn_tonm[max_Mn_idx]
    max_Pn_ton = phi_Pn_ton[max_Mn_idx]
    ax.plot(max_Mn_tonm, max_Pn_ton, 'g^', markersize=10, 
            markeredgecolor='darkgreen', markeredgewidth=2, label='Max Moment Point')
    ax.annotate(f'Max M\n({max_Mn_tonm:.1f}, {max_Pn_ton:.0f})', 
                xy=(max_Mn_tonm, max_Pn_ton),
                xytext=(15, 15), textcoords='offset points',
                fontsize=9, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1))
    
    # Plot Applied Load point (Pu, Mu) - Original load (shown as circle)
    if consider_slenderness and delta_ns > 1.0:
        # Show original load as smaller marker when magnification is used
        ax.scatter(Mu_tonm, Pu_ton, s=100, c='lightblue', marker='o', 
                   edgecolors='blue', linewidths=1.5, zorder=9, alpha=0.7,
                   label=f'Original Load (Pu, Mu)')
        ax.annotate(f'Original\n$M_u$={Mu_tonm:.2f}', 
                    xy=(Mu_tonm, Pu_ton),
                    xytext=(-40, -30), textcoords='offset points',
                    fontsize=8, color='blue', alpha=0.8,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1, alpha=0.5))
    
    # Plot Magnified Load point (Pu, Mc) when slenderness is considered
    if consider_slenderness and delta_ns > 1.0 and delta_ns != float('inf'):
        # Magnified load marker (Orange Star)
        mag_marker_color = 'green' if is_safe else 'red'
        mag_marker_face = 'orange'
        mag_status = 'SAFE ✓' if is_safe else 'UNSAFE ✗'
        
        ax.scatter(Mc_tonm, Pu_ton, s=250, c=mag_marker_face, marker='*', 
                   edgecolors='darkorange', linewidths=2, zorder=10,
                   label=f'Magnified Load (Pu, Mc) - {mag_status}')
        
        # Add annotation for magnified load
        ax.annotate(f'Magnified Load\n$P_u$ = {Pu_ton:.1f} Ton\n$M_c$ = {Mc_tonm:.2f} Ton·m\n$δ_{{ns}}$ = {delta_ns:.3f}\n({mag_status})', 
                    xy=(Mc_tonm, Pu_ton),
                    xytext=(30, 30), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='darkorange',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='darkorange', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))
        
        # Draw arrow from original to magnified load to show magnification effect
        ax.annotate('', xy=(Mc_tonm, Pu_ton), xytext=(Mu_tonm, Pu_ton),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2, 
                                   linestyle='--', alpha=0.7))
    else:
        # Plot Applied Load point (Pu, Mu) - without magnification
        marker_color = 'green' if is_safe else 'red'
        marker_face = 'lime' if is_safe else 'red'
        status_text = 'SAFE ✓' if is_safe else 'UNSAFE ✗'
        
        ax.scatter(Mu_tonm, Pu_ton, s=200, c=marker_face, marker='*', 
                   edgecolors=marker_color, linewidths=2, zorder=10,
                   label=f'Applied Load (Pu, Mu) - {status_text}')
        
        # Add annotation for applied load
        ax.annotate(f'Applied Load\n$P_u$ = {Pu_ton:.1f} Ton\n$M_u$ = {Mu_tonm:.2f} Ton·m\n({status_text})', 
                    xy=(Mu_tonm, Pu_ton),
                    xytext=(30, 30), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=marker_color,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=marker_color, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=marker_color, lw=2))
    
    # Axis labels (English only for font compatibility)
    ax.set_xlabel('Bending Moment, $φM_n$ (Ton·m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Axial Force, $φP_n$ (Ton)', fontsize=12, fontweight='bold')
    
    # Title - include slenderness info if applicable
    if consider_slenderness:
        title_text = (f'P-M Interaction Diagram\n'
                     f'Column {b_cm:.0f}×{h_cm:.0f} cm, {total_bars}-{main_bar}, '
                     f"f'c={fc_ksc:.0f} ksc, {steel_grade}\n"
                     f'Slenderness: $kL_u/r$ = {slenderness_ratio:.1f}, $δ_{{ns}}$ = {delta_ns:.3f}')
    else:
        title_text = (f'P-M Interaction Diagram\n'
                     f'Column {b_cm:.0f}×{h_cm:.0f} cm, {total_bars}-{main_bar}, '
                     f"f'c={fc_ksc:.0f} ksc, {steel_grade}")
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.3)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    # Add text box with section info
    if consider_slenderness:
        info_text = (f"Section: {b_cm:.0f}×{h_cm:.0f} cm\n"
                     f"Rebar: {total_bars}-{main_bar}\n"
                     f"f'c = {fc_ksc:.0f} ksc ({fc_mpa:.1f} MPa)\n"
                     f"fy = {fy_mpa:.0f} MPa ({steel_grade})\n"
                     f"ρ = {rho*100:.2f}%\n"
                     f"───────────\n"
                     f"Lu = {Lu_m:.2f} m, k = {k_factor:.2f}\n"
                     f"kLu/r = {slenderness_ratio:.1f}\n"
                     f"δns = {delta_ns:.3f}")
    else:
        info_text = (f"Section: {b_cm:.0f}×{h_cm:.0f} cm\n"
                     f"Rebar: {total_bars}-{main_bar}\n"
                     f"f'c = {fc_ksc:.0f} ksc ({fc_mpa:.1f} MPa)\n"
                     f"fy = {fy_mpa:.0f} MPa ({steel_grade})\n"
                     f"ρ = {rho*100:.2f}%")
    
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    # Set axis limits with some padding (include magnified moment if applicable)
    x_max_moment = max(phi_Mn_tonm.max(), Mn_tonm.max(), Mu_tonm, Mc_tonm) * 1.1
    y_max = max(phi_Pn_ton.max(), Pn_ton.max(), Pu_ton) * 1.1
    y_min = min(phi_Pn_ton.min(), Pn_ton.min(), Pu_ton, 0) * 1.1
    
    ax.set_xlim(0, x_max_moment)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # =========================================================================
    # SECOND PLOT: SI Units (kN, kN·m)
    # =========================================================================
    
    with st.expander("📊 View Plot in SI Units (kN, kN·m)"):
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
        
        # Convert applied loads to kN
        Pu_kN = Pu_ton * TON_TO_N / 1000
        Mu_kNm = Mu_tonm * TON_TO_N / 1e3
        Mc_kNm = Mc_tonm * TON_TO_N / 1e3  # Magnified moment in kN·m
        
        # Plot curves
        ax2.plot(Mn_kNm, Pn_kN, 'b--', linewidth=1.5, alpha=0.6, 
                label='Nominal Capacity (Pn, Mn)')
        ax2.plot(phi_Mn_kNm, phi_Pn_kN, 'b-', linewidth=2.5, 
                label='Design Capacity (φPn, φMn)')
        ax2.fill_between(phi_Mn_kNm, phi_Pn_kN, alpha=0.15, color='blue')
        
        # Reference lines
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
        ax2.axhline(y=pm_results['phi_Pn_max']/1000, color='red', linestyle=':', linewidth=1.5)
        
        # Key points
        ax2.plot(balanced_point['phi_Mn']/1e6, balanced_point['phi_Pn']/1000, 
                'ro', markersize=10, label='Balanced Point')
        ax2.plot(phi_Mn_kNm[max_Mn_idx], phi_Pn_kN[max_Mn_idx], 
                'g^', markersize=10, label='Max Moment Point')
        
        # Define marker variables for SI plot
        si_marker_color = 'green' if is_safe else 'red'
        si_marker_face = 'lime' if is_safe else 'red'
        si_status_text = 'SAFE ✓' if is_safe else 'UNSAFE ✗'
        
        # Applied load with magnification handling
        if consider_slenderness and delta_ns > 1.0 and delta_ns != float('inf'):
            # Original load (smaller marker)
            ax2.scatter(Mu_kNm, Pu_kN, s=100, c='lightblue', marker='o', 
                       edgecolors='blue', linewidths=1.5, zorder=9, alpha=0.7,
                       label='Original Load (Pu, Mu)')
            
            # Magnified load (orange star)
            ax2.scatter(Mc_kNm, Pu_kN, s=250, c='orange', marker='*', 
                       edgecolors='darkorange', linewidths=2, zorder=10,
                       label=f'Magnified Load (Pu, Mc) - {si_status_text}')
            ax2.annotate(f'$P_u$ = {Pu_kN:.0f} kN\n$M_c$ = {Mc_kNm:.1f} kN·m\nδns = {delta_ns:.3f}', 
                        xy=(Mc_kNm, Pu_kN), xytext=(20, 20), textcoords='offset points',
                        fontsize=10, color='darkorange', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkorange'),
                        arrowprops=dict(arrowstyle='->', color='darkorange'))
        else:
            ax2.scatter(Mu_kNm, Pu_kN, s=200, c=si_marker_face, marker='*', 
                       edgecolors=si_marker_color, linewidths=2, zorder=10,
                       label=f'Applied Load - {si_status_text}')
            ax2.annotate(f'$P_u$ = {Pu_kN:.0f} kN\n$M_u$ = {Mu_kNm:.1f} kN·m', 
                        xy=(Mu_kNm, Pu_kN), xytext=(20, 20), textcoords='offset points',
                        fontsize=10, color=si_marker_color, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor=si_marker_color),
                        arrowprops=dict(arrowstyle='->', color=si_marker_color))
        
        ax2.set_xlabel('Bending Moment, $φM_n$ (kN·m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Axial Force, $φP_n$ (kN)', fontsize=12, fontweight='bold')
        ax2.set_title('P-M Interaction Diagram (SI Units)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Show data table in expander
    with st.expander("📊 View Interaction Curve Data"):
        import pandas as pd
        
        # Create DataFrame with key points
        df = pd.DataFrame({
            'c (mm)': pm_results['c_values'],
            'εt': pm_results['epsilon_t'],
            'φ': pm_results['phi'],
            'Pn (kN)': Pn_kN,
            'Mn (kN·m)': Mn_kNm,
            'φPn (kN)': phi_Pn_kN,
            'φMn (kN·m)': phi_Mn_kNm,
            'φPn (Ton)': phi_Pn_ton,
            'φMn (Ton·m)': phi_Mn_tonm,
        })
        
        # Format the dataframe for display
        df_display = df.copy()
        df_display['c (mm)'] = df_display['c (mm)'].apply(lambda x: f'{x:.1f}')
        df_display['εt'] = df_display['εt'].apply(lambda x: f'{x:.5f}')
        df_display['φ'] = df_display['φ'].apply(lambda x: f'{x:.3f}')
        df_display['Pn (kN)'] = df_display['Pn (kN)'].apply(lambda x: f'{x:.1f}')
        df_display['Mn (kN·m)'] = df_display['Mn (kN·m)'].apply(lambda x: f'{x:.2f}')
        df_display['φPn (kN)'] = df_display['φPn (kN)'].apply(lambda x: f'{x:.1f}')
        df_display['φMn (kN·m)'] = df_display['φMn (kN·m)'].apply(lambda x: f'{x:.2f}')
        df_display['φPn (Ton)'] = df_display['φPn (Ton)'].apply(lambda x: f'{x:.2f}')
        df_display['φMn (Ton·m)'] = df_display['φMn (Ton·m)'].apply(lambda x: f'{x:.3f}')
        st.table(df_display)
    
    # Show rebar layer details
    with st.expander("🔩 View Rebar Layer Details"):
        st.markdown("**Reinforcement Layer Configuration:**")
        layers_data = []
        for i, layer in enumerate(pm_results['rebar_layers']):
            layers_data.append({
                'Layer': i + 1,
                'Distance from top (mm)': layer.distance,
                'Area (mm²)': layer.area,
                'Number of bars equiv.': layer.area / get_rebar_area(main_bar_dia_mm)
            })
        
        df_layers = pd.DataFrame(layers_data)
        # Format the dataframe for display
        df_layers_display = df_layers.copy()
        df_layers_display['Distance from top (mm)'] = df_layers_display['Distance from top (mm)'].apply(lambda x: f'{x:.1f}')
        df_layers_display['Area (mm²)'] = df_layers_display['Area (mm²)'].apply(lambda x: f'{x:.1f}')
        df_layers_display['Number of bars equiv.'] = df_layers_display['Number of bars equiv.'].apply(lambda x: f'{x:.1f}')
        st.table(df_layers_display)
    
    # =========================================================================
    # PDF EXPORT SECTION
    # =========================================================================
    
    st.markdown("---")
    st.subheader("📄 Export Report to PDF")
    
    # Prepare design inputs dictionary for PDF
    pdf_design_inputs = {
        'fc_ksc': fc_ksc,
        'fc_mpa': fc_mpa,
        'fy_mpa': fy_mpa,
        'steel_grade': steel_grade,
        'b_cm': b_cm,
        'h_cm': h_cm,
        'cover_cm': cover_cm,
        'Ag_cm2': Ag_mm2 / 100,  # Convert mm² to cm²
        'main_bar': main_bar,
        'n_bars_x': n_bars_x,
        'n_bars_y': n_bars_y,
        'total_bars': total_bars,
        'As_cm2': As_total_mm2 / 100,  # Convert mm² to cm²
        'rho_percent': rho * 100,
        'stirrup_bar': stirrup_bar,
        'stirrup_spacing_m': stirrup_spacing_m,
        'Pu_ton': Pu_ton,
        'Mu_tonm': Mu_tonm,
        'consider_slenderness': consider_slenderness,
        'Lu_m': Lu_m,
        'k_factor': k_factor,
        'delta_ns': delta_ns,
        'Mc_tonm': Mc_tonm,
        'slenderness_ratio': slenderness_ratio,
        'is_long_column': is_slender,
        'Pc_N': Pc_N,
    }
    
    # Prepare results dictionary for PDF
    # Get the maximum design moment (phi*Mn)
    phi_Mn_max = np.max(pm_results['phi_Mn'])
    
    pdf_results = {
        'Pn_max_ton': np.max(pm_results['Pn']) / TON_TO_N,
        'phi_Pn_max_ton': pm_results['phi_Pn_max'] / TON_TO_N,
        'Mn_max_tonm': np.max(pm_results['Mn']) / (TON_TO_N * 1000),
        'phi_Mn_max_tonm': phi_Mn_max / (TON_TO_N * 1000),
        'status': 'SAFE' if is_safe else 'UNSAFE',
        'dc_ratio': capacity_ratio,
        'utilization_percent': capacity_ratio * 100,
    }
    
    # PDF Export columns
    col_pdf1, col_pdf2 = st.columns([2, 3])
    
    with col_pdf1:
        st.markdown("""
        **Export Options:**
        - Full design report with all inputs and results
        - Includes Cross-Section drawing
        - Includes P-M Interaction Diagram
        - Professional format for documentation
        """)
    
    with col_pdf2:
        # Generate PDF button
        try:
            # Create the PDF
            pdf_bytes = create_pdf_report(
                design_inputs=pdf_design_inputs,
                results=pdf_results,
                pm_results=pm_results,  # Add P-M interaction data
                fig_interaction=fig,
                fig_section=fig_section,
                font_path=None  # Will try to auto-detect Thai fonts
            )
            
            # Download button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"RC_Column_Design_{b_cm:.0f}x{h_cm:.0f}cm_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                help="Download the complete design report as PDF",
                type="primary",
                use_container_width=True
            )
            
            st.success("✅ PDF generated successfully! Click button to download.")
            
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")
            st.info("Please ensure fpdf2 is installed: `pip install fpdf2`")
    
    # =========================================================================
    # CONVERSION REFERENCE TABLE
    # =========================================================================
    
    st.markdown("---")
    st.subheader("📖 Unit Conversion Reference")

    col_conv1, col_conv2, col_conv3 = st.columns(3)
    
    with col_conv1:
        st.markdown("**Length:**")
        st.write("- 1 cm = 10 mm")
        st.write("- 1 m = 100 cm = 1000 mm")
    
    with col_conv2:
        st.markdown("**Stress:**")
        st.write("- 1 ksc = 0.0981 MPa")
        st.write("- 1 MPa = 10.197 ksc")
        st.write(f"- {fc_ksc:.0f} ksc = {fc_mpa:.2f} MPa")
    
    with col_conv3:
        st.markdown("**Force:**")
        st.write("- 1 Ton = 9,806.65 N")
        st.write("- 1 Ton = 9.807 kN")
        st.write("- 1 kN = 0.102 Ton")
    
    # =========================================================================
    # STORE VALUES IN SESSION STATE FOR FURTHER CALCULATIONS
    # =========================================================================
    
    st.session_state.design_inputs = {
        # Original inputs (Thai units)
        "fc_ksc": fc_ksc,
        "fy_mpa": fy_mpa,
        "b_cm": b_cm,
        "h_cm": h_cm,
        "cover_cm": cover_cm,
        "main_bar": main_bar,
        "main_bar_dia_mm": main_bar_dia_mm,
        "stirrup_bar": stirrup_bar,
        "stirrup_dia_mm": stirrup_dia_mm,
        "stirrup_spacing_cm": stirrup_spacing_cm,
        "stirrup_spacing_m": stirrup_spacing_m,
        "fy_stirrup_mpa": fy_stirrup_mpa,
        "n_bars_x": n_bars_x,
        "n_bars_y": n_bars_y,
        "total_bars": total_bars,
        
        # Converted values (SI: N, mm, MPa)
        "fc_mpa": fc_mpa,
        "b_mm": b_mm,
        "h_mm": h_mm,
        "cover_mm": cover_mm,
        "d_mm": d_mm,
        "d_prime_mm": d_prime_mm,
        "Ag_mm2": Ag_mm2,
        "As_total_mm2": As_total_mm2,
        "rho": rho,
        
        # Applied loads
        "Pu_ton": Pu_ton,
        "Mu_tonm": Mu_tonm,
        "Pu_N": Pu_N,
        "Mu_Nmm": Mu_Nmm,
        
        # Slenderness parameters
        "consider_slenderness": consider_slenderness,
        "Lu_m": Lu_m,
        "Lu_mm": Lu_mm,
        "k_factor": k_factor,
        "beta_dns": beta_dns,
        "Cm": Cm,
        "slenderness_ratio": slenderness_ratio,
        "is_slender": is_slender,
        "Ec_mpa": Ec_mpa,
        "Pc_N": Pc_N,
        "Pc_ton": Pc_N / TON_TO_N,
        "delta_ns": delta_ns,
        "Mc_tonm": Mc_tonm,
        "Mc_Nmm": Mc_Nmm,
        
        # Capacity results
        "is_safe": is_safe,
        "capacity_ratio": capacity_ratio,
        "phi_Pn_max_ton": pm_results['phi_Pn_max'] / TON_TO_N,
        "phi_Pnt_ton": pm_results['phi_Pnt'] / TON_TO_N,
        "beta1": pm_results['beta1'],
    }
    
    # Display stored values for debugging (can be removed in production)
    with st.expander("🔧 Debug: View All Stored Values (for calculation)"):
        # Convert to JSON-serializable format
        debug_data = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                      for k, v in st.session_state.design_inputs.items() 
                      if not isinstance(v, np.ndarray)}
        st.json(debug_data)


if __name__ == "__main__":
    main()
