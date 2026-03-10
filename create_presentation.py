"""
Generate PFE Graduation Presentation - UrbanFlow: Bike Demand Prediction
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os
import json

# === Color Palette ===
PRIMARY = RGBColor(0x1B, 0x4D, 0x89)      # Deep blue
SECONDARY = RGBColor(0x2E, 0x86, 0xAB)    # Teal blue
ACCENT = RGBColor(0xE8, 0x6F, 0x2C)       # Orange
DARK = RGBColor(0x1A, 0x1A, 0x2E)         # Near black
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_BG = RGBColor(0xF0, 0xF4, 0xF8)     # Light gray-blue
GREEN = RGBColor(0x27, 0xAE, 0x60)        # Success green
LIGHT_BLUE = RGBColor(0xEB, 0xF5, 0xFB)   # Very light blue
GRAY = RGBColor(0x6C, 0x75, 0x7D)         # Text gray
DARK_TEXT = RGBColor(0x2C, 0x3E, 0x50)     # Dark text

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_background(slide, color=WHITE):
    """Fill slide background with a solid color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color=None, line_color=None, shape_type=MSO_SHAPE.RECTANGLE):
    """Add a shape to the slide."""
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.fill.solid()
        shape.line.fill.fore_color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=DARK_TEXT, bold=False, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri", line_spacing=1.2):
    """Add a text box to the slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(0)
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=16,
                    color=DARK_TEXT, font_name="Calibri", bullet_color=ACCENT):
    """Add a bulleted list."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(6)
        p.level = 0
        # Bullet
        pPr = p._pPr
        if pPr is None:
            from pptx.oxml.ns import qn
            pPr = p._p.get_or_add_pPr()
        from pptx.oxml.ns import qn
        buFont = pPr.makeelement(qn('a:buFont'), {'typeface': 'Wingdings'})
        buChar = pPr.makeelement(qn('a:buChar'), {'char': '\u006C'})  # filled circle
        buClr = pPr.makeelement(qn('a:buClr'), {})
        srgb = buClr.makeelement(qn('a:srgbClr'), {'val': '%02X%02X%02X' % (bullet_color[0] if isinstance(bullet_color, tuple) else ACCENT.red if hasattr(ACCENT, 'red') else 0xE8, 0x6F, 0x2C)})
        buClr.append(srgb)
        # Remove existing bullet elements
        for tag in ['a:buNone', 'a:buChar', 'a:buFont', 'a:buClr']:
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
        pPr.append(buClr)
        pPr.append(buChar)
    return txBox


def add_simple_bullets(slide, left, top, width, height, items, font_size=16,
                       color=DARK_TEXT, font_name="Calibri"):
    """Add a simple bulleted list with dash prefix."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        # Use a colored bullet character
        run1 = p.add_run()
        run1.text = ">>  "
        run1.font.size = Pt(font_size)
        run1.font.color.rgb = ACCENT
        run1.font.bold = True
        run1.font.name = font_name

        run2 = p.add_run()
        run2.text = item
        run2.font.size = Pt(font_size)
        run2.font.color.rgb = color
        run2.font.name = font_name

        p.space_after = Pt(8)
    return txBox


def add_section_header(slide):
    """Add a colored bar at the top of the slide."""
    add_shape(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.08), fill_color=PRIMARY)
    add_shape(slide, Inches(0), Inches(0.08), SLIDE_W, Inches(0.04), fill_color=ACCENT)


def add_slide_number(slide, num, total=26):
    """Add slide number at the bottom right."""
    add_text_box(slide, Inches(11.5), Inches(7.05), Inches(1.5), Inches(0.4),
                 f"{num} / {total}", font_size=10, color=GRAY,
                 alignment=PP_ALIGN.RIGHT)


def add_footer_bar(slide):
    """Add a footer bar."""
    add_shape(slide, Inches(0), Inches(7.2), SLIDE_W, Inches(0.3), fill_color=PRIMARY)
    add_text_box(slide, Inches(0.5), Inches(7.2), Inches(5), Inches(0.3),
                 "UrbanFlow - Bike Demand Prediction", font_size=9, color=WHITE)


def add_card(slide, left, top, width, height, title, value, subtitle="",
             title_color=GRAY, value_color=PRIMARY, bg_color=WHITE):
    """Add a metric card."""
    shape = add_shape(slide, left, top, width, height, fill_color=bg_color, line_color=RGBColor(0xDE, 0xE2, 0xE6))
    shape.line.width = Pt(1)

    add_text_box(slide, left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), Inches(0.3),
                 title, font_size=11, color=title_color, bold=False)
    add_text_box(slide, left + Inches(0.15), top + Inches(0.4), width - Inches(0.3), Inches(0.5),
                 value, font_size=24, color=value_color, bold=True)
    if subtitle:
        add_text_box(slide, left + Inches(0.15), top + Inches(0.85), width - Inches(0.3), Inches(0.3),
                     subtitle, font_size=10, color=GRAY)


def add_title_bar(slide, title, subtitle=""):
    """Add a standard title area for content slides."""
    add_section_header(slide)
    add_text_box(slide, Inches(0.8), Inches(0.3), Inches(10), Inches(0.6),
                 title, font_size=32, color=PRIMARY, bold=True)
    if subtitle:
        add_text_box(slide, Inches(0.8), Inches(0.85), Inches(10), Inches(0.4),
                     subtitle, font_size=16, color=GRAY)
    # Divider line
    add_shape(slide, Inches(0.8), Inches(1.25), Inches(2), Inches(0.04), fill_color=ACCENT)


TOTAL_SLIDES = 28

# ============================================================
# SLIDE 1: TITLE SLIDE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
add_background(slide, PRIMARY)

# Decorative shapes
add_shape(slide, Inches(0), Inches(0), Inches(4), SLIDE_H, fill_color=RGBColor(0x15, 0x3D, 0x6E))
add_shape(slide, Inches(0), Inches(5.5), SLIDE_W, Inches(2), fill_color=RGBColor(0x0F, 0x2D, 0x53))

# Title
add_text_box(slide, Inches(4.5), Inches(1.2), Inches(8), Inches(1),
             "UrbanFlow", font_size=52, color=WHITE, bold=True)
add_text_box(slide, Inches(4.5), Inches(2.2), Inches(8), Inches(0.8),
             "Urban Shared Bike Rental Demand Prediction",
             font_size=26, color=RGBColor(0xA0, 0xC4, 0xE8), bold=False)

# Accent line
add_shape(slide, Inches(4.5), Inches(3.2), Inches(3), Inches(0.06), fill_color=ACCENT)

# Subtitle info
add_text_box(slide, Inches(4.5), Inches(3.6), Inches(8), Inches(0.5),
             "Using Machine Learning & Ensemble Methods",
             font_size=20, color=WHITE)

# Author info
add_text_box(slide, Inches(4.5), Inches(5.8), Inches(8), Inches(0.4),
             "Presented by: Yousra El Harchi", font_size=18, color=WHITE, bold=True)
add_text_box(slide, Inches(4.5), Inches(6.25), Inches(8), Inches(0.4),
             "Graduation Project (PFE) | 2025-2026", font_size=14, color=RGBColor(0xA0, 0xC4, 0xE8))

# Left side label
add_text_box(slide, Inches(0.3), Inches(3), Inches(3.4), Inches(1.5),
             "PFE\nPresentation", font_size=36, color=RGBColor(0x3A, 0x7B, 0xBD),
             bold=True, alignment=PP_ALIGN.CENTER)

add_slide_number(slide, 1, TOTAL_SLIDES)


# ============================================================
# SLIDE 2: OUTLINE / AGENDA
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Presentation Outline")

sections = [
    ("01", "Introduction & Motivation"),
    ("02", "Dataset & Exploration"),
    ("03", "Feature Engineering"),
    ("04", "Machine Learning Models"),
    ("05", "Results & Evaluation"),
    ("06", "Web Application"),
    ("07", "Deployment & Testing"),
    ("08", "Conclusion & Future Work"),
]

for i, (num, title) in enumerate(sections):
    row = i // 2
    col = i % 2
    left = Inches(1 + col * 5.8)
    top = Inches(1.6 + row * 1.3)

    # Number circle
    add_shape(slide, left, top + Inches(0.05), Inches(0.55), Inches(0.55),
              fill_color=PRIMARY if i < 5 else SECONDARY, shape_type=MSO_SHAPE.OVAL)
    add_text_box(slide, left, top + Inches(0.08), Inches(0.55), Inches(0.55),
                 num, font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    # Section title
    add_text_box(slide, left + Inches(0.75), top + Inches(0.08), Inches(4.5), Inches(0.5),
                 title, font_size=20, color=DARK_TEXT, bold=True)

add_footer_bar(slide)
add_slide_number(slide, 2, TOTAL_SLIDES)


# ============================================================
# SLIDE 3: INTRODUCTION
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Introduction",
              "Project context and academic framework")

# Left side - Introduction paragraph
add_text_box(slide, Inches(0.8), Inches(1.5), Inches(6.5), Inches(5),
             "With the rapid growth of urban populations, cities face increasing challenges "
             "in transportation and sustainability. Bike-sharing systems have emerged as a key "
             "solution for reducing traffic congestion and carbon emissions, promoting greener "
             "and healthier urban mobility.\n\n"
             "However, managing these systems efficiently remains a major challenge. Stations "
             "frequently experience shortages during peak hours or overflow during off-peak times, "
             "leading to user dissatisfaction and high operational costs.\n\n"
             "This graduation project addresses this problem by applying machine learning techniques "
             "to predict hourly bike rental demand in Seoul, South Korea. By analyzing weather, "
             "temporal, and seasonal patterns from a full year of data, the project builds an "
             "intelligent prediction system deployed as a production-ready web application.",
             font_size=15, color=DARK_TEXT)

# Right side - Academic info card
add_shape(slide, Inches(8), Inches(1.5), Inches(4.5), Inches(5),
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6))

add_text_box(slide, Inches(8.3), Inches(1.7), Inches(4), Inches(0.4),
             "Academic Framework", font_size=18, color=PRIMARY, bold=True)

add_text_box(slide, Inches(8.3), Inches(2.3), Inches(4), Inches(0.3),
             "University", font_size=12, color=GRAY)
add_text_box(slide, Inches(8.3), Inches(2.6), Inches(4), Inches(0.4),
             "China Jiliang University", font_size=16, color=DARK_TEXT, bold=True)

add_text_box(slide, Inches(8.3), Inches(3.1), Inches(4), Inches(0.3),
             "Department", font_size=12, color=GRAY)
add_text_box(slide, Inches(8.3), Inches(3.4), Inches(4), Inches(0.4),
             "Computer Science", font_size=16, color=DARK_TEXT, bold=True)

add_text_box(slide, Inches(8.3), Inches(3.9), Inches(4), Inches(0.3),
             "Supervisor", font_size=12, color=GRAY)
add_text_box(slide, Inches(8.3), Inches(4.2), Inches(4), Inches(0.4),
             "Zhu Wenjie", font_size=16, color=DARK_TEXT, bold=True)

add_text_box(slide, Inches(8.3), Inches(4.7), Inches(4), Inches(0.3),
             "Student", font_size=12, color=GRAY)
add_text_box(slide, Inches(8.3), Inches(5.0), Inches(4), Inches(0.4),
             "Yousra El Harchi", font_size=16, color=DARK_TEXT, bold=True)

add_text_box(slide, Inches(8.3), Inches(5.5), Inches(4), Inches(0.3),
             "Academic Year", font_size=12, color=GRAY)
add_text_box(slide, Inches(8.3), Inches(5.8), Inches(4), Inches(0.4),
             "2025 - 2026", font_size=16, color=DARK_TEXT, bold=True)

# Bottom - project name banner
add_shape(slide, Inches(0.8), Inches(6.2), Inches(11.5), Inches(0.8),
          fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(1.1), Inches(6.25), Inches(11), Inches(0.35),
             "UrbanFlow", font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1.1), Inches(6.55), Inches(11), Inches(0.35),
             "Urban Shared Bike Rental Demand Prediction Using Machine Learning & Ensemble Methods",
             font_size=13, color=RGBColor(0xA0, 0xC4, 0xE8), alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 3, TOTAL_SLIDES)


# ============================================================
# SLIDE 4: PROBLEM STATEMENT & MOTIVATION
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Problem Statement & Motivation",
              "Challenges in urban bike-sharing and why ML is the answer")

# --- THE PROBLEM section ---
add_shape(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(0.45),
          fill_color=RGBColor(0xFD, 0xF2, 0xE9))
add_text_box(slide, Inches(1.0), Inches(1.53), Inches(5.5), Inches(0.4),
             "The Problem", font_size=18, color=ACCENT, bold=True)

problem_items = [
    "Bike demand fluctuates heavily by hour, season, and weather conditions",
    "Stations experience shortages at peak hours and overflow at off-peak",
    "Demand patterns are non-linear and involve complex feature interactions",
]
add_simple_bullets(slide, Inches(0.8), Inches(2.1), Inches(5.8), Inches(2),
                   problem_items, font_size=15)

# --- MOTIVATION section ---
add_shape(slide, Inches(0.8), Inches(3.8), Inches(5.8), Inches(0.45),
          fill_color=RGBColor(0xE8, 0xF5, 0xE9))
add_text_box(slide, Inches(1.0), Inches(3.83), Inches(5.5), Inches(0.4),
             "Motivation", font_size=18, color=GREEN, bold=True)

motivation_items = [
    "ML can capture non-linear relationships between weather, time, and demand",
    "Accurate predictions enable proactive fleet redistribution and cost savings",
    "Data-driven decisions replace manual guesswork in operations",
]
add_simple_bullets(slide, Inches(0.8), Inches(4.4), Inches(5.8), Inches(2),
                   motivation_items, font_size=15)

# --- Right side: Research Question + Objective ---
# Research question box
add_shape(slide, Inches(7.2), Inches(1.5), Inches(5.5), Inches(2.2),
          fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(7.5), Inches(1.65), Inches(5), Inches(0.4),
             "Research Question", font_size=14, color=RGBColor(0xA0, 0xC4, 0xE8))
add_text_box(slide, Inches(7.5), Inches(2.1), Inches(5), Inches(1.4),
             "How can machine learning models\naccurately predict hourly bike\nrental demand using weather,\ntemporal, and seasonal features?",
             font_size=17, color=WHITE, bold=True)

# Our approach box
add_shape(slide, Inches(7.2), Inches(4.0), Inches(5.5), Inches(2.5),
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6))
add_text_box(slide, Inches(7.5), Inches(4.15), Inches(5), Inches(0.4),
             "Our Approach", font_size=16, color=PRIMARY, bold=True)

approach_items = [
    "8,760 hourly records from Seoul (full year)",
    "36+ engineered features from 13 raw columns",
    "6 ML models including LightGBM & stacking ensemble",
    "Production-ready Flask web application",
]
add_simple_bullets(slide, Inches(7.3), Inches(4.6), Inches(5.2), Inches(2),
                   approach_items, font_size=14)

add_footer_bar(slide)
add_slide_number(slide, 4, TOTAL_SLIDES)


# ============================================================
# SLIDE 5: PROJECT OBJECTIVES
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Project Objectives")

objectives = [
    ("Data Analysis", "Explore and understand Seoul bike rental data patterns\nacross seasons, weather, and time dimensions"),
    ("ML Pipeline", "Build a leakage-free machine learning pipeline\nwith proper time-series validation"),
    ("Model Comparison", "Train and compare multiple models:\nRandom Forest, XGBoost, LightGBM, Stacking Ensemble"),
    ("Web Application", "Deploy a production-ready Flask web app\nwith real-time prediction and analytics"),
]

for i, (title, desc) in enumerate(objectives):
    top = Inches(1.6 + i * 1.35)
    # Number
    add_shape(slide, Inches(0.8), top, Inches(0.6), Inches(0.6),
              fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(0.8), top + Inches(0.05), Inches(0.6), Inches(0.5),
                 str(i + 1), font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    # Title
    add_text_box(slide, Inches(1.7), top, Inches(4), Inches(0.4),
                 title, font_size=20, color=PRIMARY, bold=True)
    # Description
    add_text_box(slide, Inches(1.7), top + Inches(0.4), Inches(10), Inches(0.9),
                 desc, font_size=15, color=DARK_TEXT)

add_footer_bar(slide)
add_slide_number(slide, 5, TOTAL_SLIDES)


# ============================================================
# SLIDE 6: DATASET OVERVIEW
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Dataset Overview",
              "Seoul Bike Sharing Dataset (Dec 2017 - Nov 2018)")

# Key metrics cards
cards_data = [
    ("RECORDS", "8,760", "Hourly observations"),
    ("FEATURES", "13", "Original columns"),
    ("PERIOD", "12 months", "Full seasonal cycle"),
    ("TARGET", "Rented Bike\nCount", "Hourly rentals"),
]
for i, (title, value, sub) in enumerate(cards_data):
    add_card(slide, Inches(0.8 + i * 3), Inches(1.6), Inches(2.7), Inches(1.2),
             title, value, sub)

# Feature list
add_text_box(slide, Inches(0.8), Inches(3.2), Inches(6), Inches(0.4),
             "Raw Features:", font_size=18, color=PRIMARY, bold=True)

features_left = [
    "Date & Hour (temporal context)",
    "Temperature & Dew Point",
    "Humidity & Wind Speed",
]
features_right = [
    "Visibility & Solar Radiation",
    "Rainfall & Snowfall",
    "Seasons, Holiday, Functioning Day",
]
add_simple_bullets(slide, Inches(0.8), Inches(3.7), Inches(5.5), Inches(2.5),
                   features_left, font_size=16)
add_simple_bullets(slide, Inches(6.5), Inches(3.7), Inches(5.5), Inches(2.5),
                   features_right, font_size=16)

# Challenge callout
challenge_box = add_shape(slide, Inches(0.8), Inches(5.8), Inches(11.5), Inches(1),
                          fill_color=LIGHT_BLUE, line_color=SECONDARY)
challenge_box.line.width = Pt(1.5)
add_text_box(slide, Inches(1.1), Inches(5.9), Inches(11), Inches(0.8),
             "Challenge: Korean character encoding in column names | Time-based split needed to prevent data leakage",
             font_size=14, color=SECONDARY, bold=True)

add_footer_bar(slide)
add_slide_number(slide, 6, TOTAL_SLIDES)


# ============================================================
# SLIDE 7: DATA EXPLORATION (now correctly numbered)
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Data Exploration & Insights",
              "Key patterns discovered in exploratory data analysis")

# --- Temporal Patterns section ---
add_text_box(slide, Inches(0.8), Inches(1.55), Inches(5), Inches(0.35),
             "Temporal Patterns", font_size=14, bold=True, color=PRIMARY)

temporal = [
    "Dual commute peaks: 8 AM (1,256 avg) and 6 PM (1,658 avg) on weekdays",
    "Weekend peak shifts to 5 PM (1,139 avg) -- leisure-driven usage",
    "Summer demand is 4.6x higher than Winter (1,034 vs 226 avg/hr)",
]
add_simple_bullets(slide, Inches(0.8), Inches(1.9), Inches(6.8), Inches(1.5),
                   temporal, font_size=14)

# --- Weather & External Factors section ---
add_text_box(slide, Inches(0.8), Inches(3.25), Inches(5), Inches(0.35),
             "Weather & External Factors", font_size=14, bold=True, color=PRIMARY)

weather = [
    "Temperature is #1 predictor (r = 0.54) -- demand peaks at 20-30 C (1,120 avg)",
    "Rain drops demand by 78%: only 163 avg vs 739 on dry hours",
    "Non-functioning days = exactly 0 rentals (295 hours, 3.4% of data)",
    "Holidays reduce demand by 30% (500 vs 715 avg bikes/hr)",
]
add_simple_bullets(slide, Inches(0.8), Inches(3.6), Inches(6.8), Inches(1.8),
                   weather, font_size=14)

# --- Distribution note ---
add_text_box(slide, Inches(0.8), Inches(5.2), Inches(6.8), Inches(0.5),
             "Distribution: Highly right-skewed (skewness = 1.15) -- median 504 vs mean 705",
             font_size=12, color=RGBColor(0x60, 0x60, 0x60))

# Key numbers on right
add_card(slide, Inches(8.2), Inches(1.6), Inches(4.1), Inches(1.0),
         "RECORD HIGH", "3,556", "Maximum bikes rented in a single hour",
         value_color=ACCENT)
add_card(slide, Inches(8.2), Inches(2.8), Inches(4.1), Inches(1.0),
         "SUMMER EVENINGS", "1,850", "Average demand 5-7 PM in summer",
         value_color=RGBColor(0xF5, 0x9E, 0x0B))
add_card(slide, Inches(8.2), Inches(4.0), Inches(4.1), Inches(1.0),
         "OVERALL AVERAGE", "705", "Bikes rented per hour (all conditions)",
         value_color=GREEN)
add_card(slide, Inches(8.2), Inches(5.2), Inches(4.1), Inches(1.0),
         "ZERO-DEMAND HOURS", "295", "3.4% of hours with no rentals at all",
         value_color=PRIMARY)

add_footer_bar(slide)
add_slide_number(slide, 7, TOTAL_SLIDES)


# ============================================================
# SLIDE 8: DATA PREPROCESSING
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Data Preprocessing",
              "Cleaning and preparing data for modeling")

# --- Left column: Pipeline Steps (numbered) ---
steps = [
    ("1. Encoding Fix",
     "Korean character (U+CE98) in CSV headers resolved via multi-encoding\n"
     "fallback (UTF-8, Latin-1, CP1252). Column names standardized."),
    ("2. Chronological Split",
     "80/20 time-based split: train = first 7,008 hours, test = last 1,752 hours.\n"
     "Data sorted by (Date, Hour) -- no random shuffle, simulates real deployment."),
    ("3. Missing Value Imputation",
     "Lag features filled with training set median only (no forward-fill).\n"
     "Remaining NaNs set to 0. Test set uses same train-derived medians."),
    ("4. Feature Scaling",
     "StandardScaler fitted on training data exclusively.\n"
     "Binary columns (0/1 flags) are auto-detected and excluded from scaling."),
    ("5. Categorical Encoding",
     "One-hot encoding: Seasons (4), Holiday (2), Functioning Day (2).\n"
     "Cyclical encoding: Hour (sin/cos), Month (sin/cos), Day of Week (sin/cos)."),
]

for i, (title, desc) in enumerate(steps):
    top = Inches(1.45 + i * 0.95)
    # Step number circle
    add_shape(slide, Inches(0.6), top + Inches(0.05), Inches(0.12), Inches(0.12),
              fill_color=ACCENT, shape_type=MSO_SHAPE.OVAL)
    add_text_box(slide, Inches(1.0), top, Inches(3.5), Inches(0.3),
                 title, font_size=15, color=PRIMARY, bold=True)
    add_text_box(slide, Inches(1.0), top + Inches(0.3), Inches(5.8), Inches(0.6),
                 desc, font_size=12, color=DARK_TEXT)

# --- Right column: Leakage-Free Design callout ---
leak_box = add_shape(slide, Inches(7.5), Inches(1.5), Inches(5), Inches(2.2),
                     fill_color=RGBColor(0xFD, 0xF2, 0xE9), line_color=ACCENT)
leak_box.line.width = Pt(2)
add_text_box(slide, Inches(7.8), Inches(1.6), Inches(4.5), Inches(0.35),
             "Leakage-Free Design", font_size=16, color=ACCENT, bold=True)
leak_points = [
    "Scaler fitted on train set only",
    "Lag features use shift() to prevent future data access",
    "Median imputation computed from train only",
    "TimeSeriesSplit (5-fold) for cross-validation",
]
add_simple_bullets(slide, Inches(7.8), Inches(2.0), Inches(4.4), Inches(1.5),
                   leak_points, font_size=12)

# --- Right column: Feature dimensions card ---
add_card(slide, Inches(7.5), Inches(4.0), Inches(2.4), Inches(1.0),
         "ORIGINAL", "14", "features (raw + one-hot)",
         value_color=PRIMARY)
add_card(slide, Inches(10.1), Inches(4.0), Inches(2.4), Inches(1.0),
         "ADVANCED", "36", "features (engineered)",
         value_color=ACCENT)

# --- Bottom right: Validation note ---
add_text_box(slide, Inches(7.5), Inches(5.3), Inches(5), Inches(0.5),
             "Cross-validation: TimeSeriesSplit (5 folds) respects temporal order",
             font_size=11, color=RGBColor(0x60, 0x60, 0x60))

add_footer_bar(slide)
add_slide_number(slide, 8, TOTAL_SLIDES)


# ============================================================
# SLIDE 9: FEATURE ENGINEERING (PART 1)
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Feature Engineering (1/2)",
              "Temporal, polynomial, and weather interaction features")

# --- Left column: Temporal & Polynomial ---
header_box = add_shape(slide, Inches(0.6), Inches(1.5), Inches(5.8), Inches(0.45),
                       fill_color=PRIMARY)
add_text_box(slide, Inches(0.8), Inches(1.52), Inches(5.5), Inches(0.4),
             "Temporal & Polynomial Features", font_size=16, color=WHITE, bold=True)

temporal_items = [
    "hour_sin, hour_cos -- cyclical encoding (no 23-to-0 discontinuity)",
    "hour_squared, hour_cubed -- polynomial non-linear patterns",
    "day_of_week, is_weekend -- weekly seasonality",
    "is_peak_hour -- 1 if hour in [7-9] or [17-19]",
    "rush_hour_intensity -- graded: 1.0 / 0.6 / 0.3 / 0.0 by hour",
    "is_daylight, workday_hour_interaction",
]
add_simple_bullets(slide, Inches(0.8), Inches(2.1), Inches(5.5), Inches(2.8),
                   temporal_items, font_size=13)

# --- Right column: Weather & Interactions ---
header_box2 = add_shape(slide, Inches(6.7), Inches(1.5), Inches(5.8), Inches(0.45),
                        fill_color=PRIMARY)
add_text_box(slide, Inches(6.9), Inches(1.52), Inches(5.5), Inches(0.4),
             "Weather & Interaction Features", font_size=16, color=WHITE, bold=True)

weather_items = [
    "temp_hour_interaction -- temperature effect varies by hour",
    "temp_humidity_interaction -- perceived comfort",
    "temp_squared -- captures non-linear temp response",
    "wind_chill = 13.12 + 0.62T - 11.37(WS^0.16) + ...",
    "comfort_index = temperature - humidity / 10",
    "weather_severity_score -- composite: temp + rain + snow + wind",
    "good_biking_conditions -- boolean (10-30 C, no rain/snow, low wind)",
]
add_simple_bullets(slide, Inches(6.9), Inches(2.1), Inches(5.5), Inches(3.2),
                   weather_items, font_size=13)

# --- Bottom: Feature count summary ---
add_shape(slide, Inches(0.6), Inches(5.2), Inches(11.9), Inches(0.9),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.9), Inches(5.25), Inches(11.5), Inches(0.8),
             "13 raw features  -->  36 features (Advanced / XGBoost / LightGBM)  |  38 features (Improved RF)\n"
             "Each model tier uses a different subset of engineered features optimized for its architecture",
             font_size=13, color=PRIMARY, bold=True, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 9, TOTAL_SLIDES)


# ============================================================
# SLIDE 10: FEATURE ENGINEERING (PART 2)
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Feature Engineering (2/2)",
              "Lag features, categorical encodings, and feature summary")

# --- Left column: Lag Features ---
header_lag = add_shape(slide, Inches(0.6), Inches(1.5), Inches(5.8), Inches(0.45),
                       fill_color=SECONDARY)
add_text_box(slide, Inches(0.8), Inches(1.52), Inches(5.5), Inches(0.4),
             "Lag Features (Past Demand History)", font_size=16, color=WHITE, bold=True)

lag_items = [
    "demand_lag_24h -- same hour yesterday (shift 24)",
    "demand_lag_168h -- same hour last week (shift 168)",
    "demand_rolling_3h -- 3-hour rolling mean (shift 1 first)",
    "demand_rolling_24h -- 24-hour rolling mean (shift 1 first)",
    "temp_deviation_from_season -- temperature anomaly vs seasonal norm",
]
add_simple_bullets(slide, Inches(0.8), Inches(2.1), Inches(5.5), Inches(2.2),
                   lag_items, font_size=13)

# Leakage prevention note under lag features
leak_note = add_shape(slide, Inches(0.6), Inches(4.0), Inches(5.8), Inches(1.0),
                      fill_color=RGBColor(0xFD, 0xF2, 0xE9), line_color=ACCENT)
leak_note.line.width = Pt(1.5)
add_text_box(slide, Inches(0.85), Inches(4.05), Inches(5.3), Inches(0.9),
             "Leakage prevention: shift(1) applied BEFORE rolling\n"
             "window so current hour is never included.\n"
             "NaN values filled with training-set median only.",
             font_size=12, color=DARK_TEXT)

# --- Right column: Categorical Encodings ---
header_cat = add_shape(slide, Inches(6.7), Inches(1.5), Inches(5.8), Inches(0.45),
                       fill_color=SECONDARY)
add_text_box(slide, Inches(6.9), Inches(1.52), Inches(5.5), Inches(0.4),
             "Categorical Encodings", font_size=16, color=WHITE, bold=True)

cat_items = [
    "Seasons -- one-hot: Spring, Summer, Autumn, Winter (4 cols)",
    "Holiday -- one-hot: Holiday, No Holiday (2 cols)",
    "Functioning Day -- one-hot: Yes, No (2 cols)",
    "temp_category -- binned: freezing / cold / cool / warm / hot (5 cols)",
    "time_of_day -- binned: morning / afternoon / evening / night (4 cols)",
    "weather_category -- binned: severe / poor / good / perfect (4 cols)",
]
add_simple_bullets(slide, Inches(6.9), Inches(2.1), Inches(5.5), Inches(2.5),
                   cat_items, font_size=13)

# --- Bottom: Feature comparison table ---
add_text_box(slide, Inches(6.7), Inches(4.6), Inches(5.8), Inches(0.35),
             "Features by Model Tier:", font_size=14, color=PRIMARY, bold=True)

tier_items = [
    "Original RF: 14 features (9 raw + 5 one-hot)",
    "Improved RF: 38 features (+ lags, indices, weather categories)",
    "Advanced / XGBoost / LightGBM: 36 features (+ cyclical, temp bins)",
    "Stacking Ensemble: 36 features (same as its base learners)",
]
add_simple_bullets(slide, Inches(6.9), Inches(4.9), Inches(5.5), Inches(1.5),
                   tier_items, font_size=12)

add_footer_bar(slide)
add_slide_number(slide, 10, TOTAL_SLIDES)


# ============================================================
# SLIDE 11: ML MODELS OVERVIEW
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Machine Learning Models",
              "6 models across 4 algorithm families")

# Column headers
add_text_box(slide, Inches(0.6), Inches(1.35), Inches(2.4), Inches(0.3),
             "Algorithm", font_size=12, color=RGBColor(0x60, 0x60, 0x60), bold=True)
add_text_box(slide, Inches(3.2), Inches(1.35), Inches(3.5), Inches(0.3),
             "How It Works", font_size=12, color=RGBColor(0x60, 0x60, 0x60), bold=True)
add_text_box(slide, Inches(6.9), Inches(1.35), Inches(3.2), Inches(0.3),
             "Key Hyperparameters", font_size=12, color=RGBColor(0x60, 0x60, 0x60), bold=True)
add_text_box(slide, Inches(10.3), Inches(1.35), Inches(2.2), Inches(0.3),
             "Features", font_size=12, color=RGBColor(0x60, 0x60, 0x60), bold=True)

models_info = [
    ("Random Forest",
     PRIMARY,
     "Bagging: trains 200 independent\ndecision trees on bootstrap samples,\naverages predictions to reduce variance.",
     "n_estimators = 200\nmax_depth = None (unlimited)\nmin_samples_split = 2 / 5",
     "3 variants:\n14 / 38 / 36"),
    ("XGBoost",
     ACCENT,
     "Gradient boosting: builds 500 trees\nsequentially, each correcting errors\nof previous. L1/L2 regularization.",
     "n_estimators = 500\nmax_depth = 6, lr = 0.05\nsubsample = 0.8, colsample = 0.8",
     "36 features\nR2 = 0.944"),
    ("LightGBM",
     SECONDARY,
     "Leaf-wise growth (vs level-wise):\nexpands the leaf with max loss reduction.\nHistogram-based splitting for speed.",
     "n_estimators = 500, leaves = 31\nmax_depth = 6, lr = 0.05\nsubsample = 0.8, colsample = 0.8",
     "36 features\nR2 = 0.757"),
    ("Stacking\nEnsemble",
     GREEN,
     "Meta-learning: RF + XGBoost + LightGBM\nas base learners. RidgeCV combines their\npredictions (5-fold cross-validation).",
     "Base: RF(200) + XGB(500) + LGB(500)\nMeta: RidgeCV (auto alpha)\nCV = 5 folds",
     "36 features\nR2 = 0.791"),
]

for i, (name, color, desc, params, feats) in enumerate(models_info):
    top = Inches(1.65 + i * 1.15)
    # Model name badge
    add_shape(slide, Inches(0.6), top, Inches(2.4), Inches(1.0), fill_color=color,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(0.6), top + Inches(0.25), Inches(2.4), Inches(0.5),
                 name, font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    # How it works
    add_text_box(slide, Inches(3.2), top + Inches(0.02), Inches(3.5), Inches(0.95),
                 desc, font_size=12, color=DARK_TEXT)
    # Key hyperparameters
    add_text_box(slide, Inches(6.9), top + Inches(0.02), Inches(3.2), Inches(0.95),
                 params, font_size=11, color=RGBColor(0x40, 0x40, 0x40))
    # Features
    add_text_box(slide, Inches(10.3), top + Inches(0.02), Inches(2.2), Inches(0.95),
                 feats, font_size=12, color=PRIMARY, bold=True, alignment=PP_ALIGN.CENTER)

# Bottom note
add_shape(slide, Inches(0.6), Inches(5.95), Inches(11.9), Inches(0.55),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.9), Inches(5.98), Inches(11.5), Inches(0.5),
             "Random Forest variants: Original (14 raw features)  |  Improved (38, with lags & weather indices)  |  Advanced (36, optimized feature set)",
             font_size=12, color=PRIMARY, bold=True, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 11, TOTAL_SLIDES)


# ============================================================
# SLIDE 12: TRAINING PIPELINE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Unified Training Pipeline",
              "Leakage-free, reproducible training workflow")

# Pipeline steps as a flow
steps_flow = [
    ("1. Load Data", "Seoul bike CSV\n8,760 records"),
    ("2. Preprocess", "Encoding fix\nHandle missing"),
    ("3. Time Split", "80/20 chrono\n7,008 / 1,752"),
    ("4. Engineer", "36+ features\nLag, weather, time"),
    ("5. Scale", "StandardScaler\nTrain-only fit"),
    ("6. Train", "6 models\nTimeSeriesSplit CV"),
]

for i, (step, desc) in enumerate(steps_flow):
    left = Inches(0.5 + i * 2.1)
    top = Inches(1.6)
    color = PRIMARY if i % 2 == 0 else SECONDARY
    add_shape(slide, left, top, Inches(1.9), Inches(0.5), fill_color=color,
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, left, top + Inches(0.08), Inches(1.9), Inches(0.4),
                 step, font_size=13, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, left, top + Inches(0.6), Inches(1.9), Inches(0.7),
                 desc, font_size=11, color=DARK_TEXT, alignment=PP_ALIGN.CENTER)
    # Arrow
    if i < len(steps_flow) - 1:
        add_text_box(slide, left + Inches(1.85), top + Inches(0.05), Inches(0.3), Inches(0.4),
                     "->", font_size=16, color=ACCENT, bold=True, alignment=PP_ALIGN.CENTER)

# Cross-validation section
add_text_box(slide, Inches(0.8), Inches(3.2), Inches(6), Inches(0.4),
             "Cross-Validation: TimeSeriesSplit (5 folds)", font_size=18, color=PRIMARY, bold=True)

cv_items = [
    "Respects temporal ordering of data (no future leakage)",
    "Fold 1: Train on months 1-2, test on months 3-4",
    "Each subsequent fold uses more training data",
    "R2 improves with more data: 0.29 -> 0.49 -> 0.70 -> 0.61 -> 0.64",
    "Demonstrates model learning improves with additional data",
]
add_simple_bullets(slide, Inches(0.8), Inches(3.7), Inches(11), Inches(3),
                   cv_items, font_size=15)

add_footer_bar(slide)
add_slide_number(slide, 12, TOTAL_SLIDES)


# ============================================================
# SLIDE 13: STACKING ENSEMBLE ARCHITECTURE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Stacking Ensemble Architecture",
              "Two-level meta-learning: combining 3 base learners via RidgeCV")

# --- Input features box (left) ---
add_shape(slide, Inches(0.4), Inches(1.8), Inches(1.6), Inches(3.4),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.4), Inches(1.9), Inches(1.6), Inches(0.35),
             "Input", font_size=13, color=PRIMARY, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), Inches(2.25), Inches(1.6), Inches(2.8),
             "36 engineered\nfeatures\n\nTemporal\nWeather\nCyclical\nCategorical",
             font_size=11, color=DARK_TEXT, alignment=PP_ALIGN.CENTER)

# Arrow from input to base learners
add_text_box(slide, Inches(2.0), Inches(3.1), Inches(0.5), Inches(0.4),
             "->", font_size=18, color=GRAY, bold=True)

# --- Base Learners (Level 0) ---
add_text_box(slide, Inches(2.5), Inches(1.45), Inches(3.5), Inches(0.3),
             "Level 0: Base Learners", font_size=13, color=GRAY, bold=True,
             alignment=PP_ALIGN.CENTER)

base_models = [
    ("Random Forest", "n=200, depth=None", PRIMARY),
    ("XGBoost", "n=500, depth=6, lr=0.05", ACCENT),
    ("LightGBM", "n=500, leaves=31, lr=0.05", SECONDARY),
]

for i, (name, params, color) in enumerate(base_models):
    top = Inches(1.8 + i * 1.15)
    add_shape(slide, Inches(2.5), top, Inches(3.5), Inches(0.95),
              fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(2.5), top + Inches(0.15), Inches(3.5), Inches(0.35),
                 name, font_size=16, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(2.5), top + Inches(0.5), Inches(3.5), Inches(0.35),
                 params, font_size=11, color=RGBColor(0xE0, 0xE8, 0xF0), alignment=PP_ALIGN.CENTER)

# Arrows from base learners to predictions
for i in range(3):
    top = Inches(2.15 + i * 1.15)
    add_text_box(slide, Inches(6.1), top, Inches(0.5), Inches(0.4),
                 "->", font_size=16, color=GRAY, bold=True)

# --- Out-of-Fold Predictions box ---
add_shape(slide, Inches(6.6), Inches(1.8), Inches(2.3), Inches(3.4),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(6.6), Inches(1.85), Inches(2.3), Inches(0.35),
             "Out-of-Fold", font_size=13, color=PRIMARY, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(6.6), Inches(2.15), Inches(2.3), Inches(0.3),
             "Predictions (5-fold CV)", font_size=11, color=PRIMARY, alignment=PP_ALIGN.CENTER)

# Individual prediction labels
pred_labels = [
    ("RF preds", PRIMARY),
    ("XGB preds", ACCENT),
    ("LGBM preds", SECONDARY),
]
for i, (label, color) in enumerate(pred_labels):
    top = Inches(2.7 + i * 0.75)
    add_shape(slide, Inches(6.9), top, Inches(1.7), Inches(0.5),
              fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(6.9), top + Inches(0.08), Inches(1.7), Inches(0.35),
                 label, font_size=12, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Arrow to meta-learner
add_text_box(slide, Inches(9.0), Inches(3.1), Inches(0.5), Inches(0.4),
             "->", font_size=18, color=ACCENT, bold=True)

# --- Meta-Learner (Level 1) ---
add_text_box(slide, Inches(9.5), Inches(1.45), Inches(3.1), Inches(0.3),
             "Level 1: Meta-Learner", font_size=13, color=GRAY, bold=True,
             alignment=PP_ALIGN.CENTER)
add_shape(slide, Inches(9.5), Inches(1.8), Inches(3.1), Inches(3.4),
          fill_color=GREEN, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(9.5), Inches(2.2), Inches(3.1), Inches(0.4),
             "RidgeCV", font_size=24, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(9.5), Inches(2.7), Inches(3.1), Inches(1.8),
             "Linear regression with\nL2 regularization\n\nLearns optimal weights\nfor each base learner\n\nAlpha selected via\ncross-validation",
             font_size=11, color=RGBColor(0xD5, 0xF5, 0xE3), alignment=PP_ALIGN.CENTER)

# --- Bottom: How it works + metrics ---
add_shape(slide, Inches(0.4), Inches(5.55), Inches(6.2), Inches(1.0),
          fill_color=RGBColor(0xFD, 0xF2, 0xE9), line_color=ACCENT)
add_text_box(slide, Inches(0.6), Inches(5.6), Inches(5.8), Inches(0.3),
             "How Stacking Works", font_size=13, color=ACCENT, bold=True)
add_text_box(slide, Inches(0.6), Inches(5.9), Inches(5.8), Inches(0.6),
             "Each base learner generates out-of-fold predictions via 5-fold CV.\n"
             "These 3 prediction columns become features for RidgeCV,\n"
             "which learns the optimal blend weights to minimize error.",
             font_size=11, color=DARK_TEXT)

# Stacking metrics card
add_card(slide, Inches(6.9), Inches(5.55), Inches(2.7), Inches(1.0),
         "STACKING R2", "0.791", "Test set (chronological split)",
         value_color=GREEN)
add_card(slide, Inches(9.8), Inches(5.55), Inches(2.7), Inches(1.0),
         "STACKING RMSE", "280.5", "vs XGBoost standalone: 153.3",
         value_color=ACCENT)

add_footer_bar(slide)
add_slide_number(slide, 13, TOTAL_SLIDES)


# ============================================================
# SLIDE 14: MODEL RESULTS COMPARISON
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Model Performance Comparison",
              "All 6 models evaluated on chronological 20% test set (1,752 hours)")

# Table header
header_y = Inches(1.45)
cols = [
    (Inches(0.5), Inches(3.2), "Model"),
    (Inches(3.7), Inches(1.5), "R2 Score"),
    (Inches(5.2), Inches(1.5), "RMSE"),
    (Inches(6.7), Inches(1.5), "MAE"),
    (Inches(8.2), Inches(1.3), "Features"),
    (Inches(9.5), Inches(1.5), "Algorithm"),
]

# Header row
add_shape(slide, Inches(0.5), header_y, Inches(10.5), Inches(0.5), fill_color=PRIMARY)
for left, width, text in cols:
    add_text_box(slide, left, header_y + Inches(0.07), width, Inches(0.35),
                 text, font_size=13, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Load actual metrics from JSON files
_lgbm_r2, _lgbm_rmse, _lgbm_mae = "0.757", "302.6", "225.7"
_stack_r2, _stack_rmse, _stack_mae = "0.791", "280.5", "207.6"
_adv_r2, _adv_rmse, _adv_mae = "0.911", "192.4", "116.6"
_xgb_r2, _xgb_rmse, _xgb_mae = "0.944", "153.3", "90.9"

for _mfile, _setter in [
    ('models/lightgbm_metrics.json', 'lgbm'),
    ('models/stacking_metrics.json', 'stack'),
]:
    if os.path.exists(_mfile):
        with open(_mfile) as _f:
            _m = json.load(_f)
            _t = _m.get('test', {})
            if _setter == 'lgbm':
                _lgbm_r2 = f"{_t['r2']:.3f}"
                _lgbm_rmse = f"{_t['rmse']:.1f}"
                _lgbm_mae = f"{_t['mae']:.1f}"
            else:
                _stack_r2 = f"{_t['r2']:.3f}"
                _stack_rmse = f"{_t['rmse']:.1f}"
                _stack_mae = f"{_t['mae']:.1f}"

if os.path.exists('models/advanced_model_info.json'):
    with open('models/advanced_model_info.json') as _f:
        _p = json.load(_f)['performance']
        _adv_r2 = f"{_p['test_r2']:.3f}"
        _adv_rmse = f"{_p['test_rmse']:.1f}"
        _adv_mae = f"{_p['test_mae']:.1f}"

if os.path.exists('models/xgboost_metrics.json'):
    with open('models/xgboost_metrics.json') as _f:
        _p = json.load(_f)['performance']
        _xgb_r2 = f"{_p['test_r2']:.3f}"
        _xgb_rmse = f"{_p['test_rmse']:.1f}"
        _xgb_mae = f"{_p['test_mae']:.1f}"

# Model, R2, RMSE, MAE, Features, Algorithm, bg_color
rows_data = [
    ("Original RF (Baseline)", "0.644", "365.9", "256.4", "14", "Random Forest", LIGHT_BG),
    ("Improved RF",            "0.899", "194.6", "130.4", "38", "Random Forest", WHITE),
    ("Advanced RF",            _adv_r2, _adv_rmse, _adv_mae, "36", "Random Forest", LIGHT_BG),
    ("XGBoost",                _xgb_r2, _xgb_rmse, _xgb_mae, "36", "Gradient Boost", WHITE),
    ("LightGBM",               _lgbm_r2, _lgbm_rmse, _lgbm_mae, "36", "Gradient Boost", LIGHT_BG),
    ("Stacking Ensemble",      _stack_r2, _stack_rmse, _stack_mae, "36", "Meta-learning", WHITE),
]

# Find best R2
_best_r2 = max(float(r[1]) for r in rows_data)
_best_idx = next(i for i, r in enumerate(rows_data) if float(r[1]) == _best_r2)

for i, (model, r2, rmse, mae, feats, algo, bg) in enumerate(rows_data):
    row_y = Inches(1.95 + i * 0.47)
    bg_actual = RGBColor(0xE8, 0xF5, 0xE9) if i == _best_idx else bg
    add_shape(slide, Inches(0.5), row_y, Inches(10.5), Inches(0.47), fill_color=bg_actual)

    values = [model, r2, rmse, mae, feats, algo]
    for j, (left, width, _) in enumerate(cols):
        is_best_r2 = (j == 1 and i == _best_idx)
        c = GREEN if is_best_r2 else DARK_TEXT
        b = (i == _best_idx)
        add_text_box(slide, left, row_y + Inches(0.06), width, Inches(0.35),
                     values[j], font_size=12, color=c, bold=b,
                     alignment=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)

    # Best model indicator
    if i == _best_idx:
        add_text_box(slide, Inches(11.1), row_y + Inches(0.03), Inches(1.2), Inches(0.42),
                     "BEST", font_size=12, color=GREEN, bold=True, alignment=PP_ALIGN.CENTER)

# --- Key Insights box ---
add_shape(slide, Inches(0.5), Inches(4.85), Inches(11.5), Inches(1.7),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.8), Inches(4.9), Inches(11), Inches(0.35),
             "Key Insights", font_size=15, color=PRIMARY, bold=True)

insights = [
    "XGBoost achieves the best test performance (R2 = 0.944, RMSE = 153.3)",
    "Feature engineering impact: Original RF (0.644) -> Advanced RF (0.911) = +41% R2 improvement",
    "Gradient boosting (XGBoost 0.944) outperforms bagging (Advanced RF 0.911) on same 36 features",
    "LightGBM and Stacking trained separately with different pipeline -- lower scores reflect training setup differences",
]
add_simple_bullets(slide, Inches(0.8), Inches(5.25), Inches(11), Inches(1.2),
                   insights, font_size=12)

add_footer_bar(slide)
add_slide_number(slide, 14, TOTAL_SLIDES)


# ============================================================
# SLIDE 15: FEATURE IMPORTANCE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Feature Importance Analysis",
              "Which features drive predictions across different model architectures")

# --- Load feature importance from JSON files ---
_xgb_fi = []
if os.path.exists('models/xgboost_metrics.json'):
    with open('models/xgboost_metrics.json') as _f:
        _raw = json.load(_f)['feature_importance']
        _xgb_fi = sorted(_raw, key=lambda x: x['importance'], reverse=True)[:10]

_adv_fi = []
if os.path.exists('models/advanced_model_info.json'):
    with open('models/advanced_model_info.json') as _f:
        _raw = json.load(_f)['feature_importance']
        _adv_fi = sorted(_raw, key=lambda x: x['importance'], reverse=True)[:5]

_imp_fi = []
if os.path.exists('models/improved_model_metrics.json'):
    with open('models/improved_model_metrics.json') as _f:
        _raw = json.load(_f)['feature_importance']
        _feats = _raw['feature']
        _imps = _raw['importance']
        _pairs = [(v, _imps[k]) for k, v in _feats.items()]
        _imp_fi = sorted(_pairs, key=lambda x: x[1], reverse=True)[:5]

# Readable feature names
_feat_labels = {
    'functioning_day_No': 'Functioning Day (No)',
    'functioning_day_Yes': 'Functioning Day (Yes)',
    'temp_category_warm': 'Temp Category (Warm)',
    'time_of_day_evening': 'Time of Day (Evening)',
    'temp_hour_interaction': 'Temp x Hour Interaction',
    'season_Winter': 'Season (Winter)',
    'hour_squared': 'Hour Squared',
    'good_biking_conditions': 'Good Biking Conditions',
    'hour_sin': 'Hour (sine)',
    'rainfall': 'Rainfall',
    'temperature': 'Temperature',
    'hour': 'Hour',
    'temp_squared': 'Temp Squared',
    'Humidity%': 'Humidity',
    'demand_rolling_3h': 'Rolling 3h Demand',
    'demand_lag_168': 'Lag 168h (Last Week)',
    'demand_lag_24': 'Lag 24h (Yesterday)',
    'comfort_index': 'Comfort Index',
    'demand_rolling_24h': 'Rolling 24h Demand',
}

# --- Left: XGBoost Top 10 bar chart ---
add_text_box(slide, Inches(0.3), Inches(1.4), Inches(7), Inches(0.35),
             "XGBoost (Best Model, R2 = 0.944)", font_size=14, color=PRIMARY, bold=True)

_max_bar = Inches(3.5)
if _xgb_fi:
    _max_val = _xgb_fi[0]['importance']
    for i, item in enumerate(_xgb_fi):
        top = Inches(1.8 + i * 0.44)
        _imp = item['importance']
        _name = _feat_labels.get(item['feature'], item['feature'])
        _bw = int(_max_bar * (_imp / _max_val))

        add_text_box(slide, Inches(0.1), top, Inches(3.0), Inches(0.38),
                     _name, font_size=11, color=DARK_TEXT, alignment=PP_ALIGN.RIGHT)
        _bc = PRIMARY if i < 4 else SECONDARY
        add_shape(slide, Inches(3.2), top + Inches(0.04), _bw, Inches(0.3),
                  fill_color=_bc, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text_box(slide, Inches(3.2) + _bw + Inches(0.08), top, Inches(0.8), Inches(0.38),
                     f"{_imp*100:.1f}%", font_size=10, color=ACCENT, bold=True)

# --- Right: Cross-model comparison ---
add_text_box(slide, Inches(7.5), Inches(1.4), Inches(5), Inches(0.35),
             "Top 5 by Model (Comparison)", font_size=14, color=PRIMARY, bold=True)

# Advanced RF top 5
add_text_box(slide, Inches(7.5), Inches(1.85), Inches(5), Inches(0.3),
             "Advanced RF (R2 = 0.911)", font_size=11, color=ACCENT, bold=True)
if _adv_fi:
    _adv_items = []
    for item in _adv_fi:
        _name = _feat_labels.get(item['feature'], item['feature'])
        _adv_items.append(f"{_name} ({item['importance']*100:.1f}%)")
    add_simple_bullets(slide, Inches(7.5), Inches(2.1), Inches(5), Inches(1.5),
                       _adv_items, font_size=10)

# Improved RF top 5
add_text_box(slide, Inches(7.5), Inches(3.6), Inches(5), Inches(0.3),
             "Improved RF (R2 = 0.899)", font_size=11, color=ACCENT, bold=True)
if _imp_fi:
    _imp_items = []
    for feat, imp in _imp_fi:
        _name = _feat_labels.get(feat, feat)
        _imp_items.append(f"{_name} ({imp*100:.1f}%)")
    add_simple_bullets(slide, Inches(7.5), Inches(3.85), Inches(5), Inches(1.5),
                       _imp_items, font_size=10)

# --- Bottom: Cross-model insights ---
add_shape(slide, Inches(0.3), Inches(6.2), Inches(12.3), Inches(0.9),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.6), Inches(6.25), Inches(12), Inches(0.3),
             "Cross-Model Observations", font_size=13, color=PRIMARY, bold=True)
_obs = [
    "XGBoost leverages one-hot features (functioning_day 23.6% combined, temp_category, time_of_day)",
    "Advanced RF relies on continuous interactions (temp_hour_interaction 16.6% is #1)",
    "Improved RF is dominated by lag features (rolling_3h alone = 27.6%) -- past demand is the strongest predictor",
]
add_simple_bullets(slide, Inches(0.6), Inches(6.5), Inches(11.8), Inches(0.6),
                   _obs, font_size=10)

add_footer_bar(slide)
add_slide_number(slide, 15, TOTAL_SLIDES)


# ============================================================
# SLIDE 16: SHAP EXPLAINABILITY
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Model Explainability with SHAP",
              "SHapley Additive exPlanations -- XGBoost (best model, R2 = 0.944)")

# --- Load SHAP data from JSON ---
_shap_data = None
if os.path.exists('models/shap_analysis.json'):
    with open('models/shap_analysis.json') as _f:
        _shap_data = json.load(_f)

_shap_labels = {
    'Humidity%': 'Humidity',
    'temp_hour_interaction': 'Temp x Hour',
    'functioning_day_No': 'Non-Functioning Day',
    'Solar_Radiation_MJ/m2': 'Solar Radiation',
    'dew_point': 'Dew Point Temp',
    'rainfall': 'Rainfall',
    'Wind_speed_m/s': 'Wind Speed',
    'day_of_week': 'Day of Week',
    'hour_sin': 'Hour (sine)',
    'temperature': 'Temperature',
    'functioning_day_Yes': 'Functioning Day',
    'season_Summer': 'Season (Summer)',
    'holiday_Holiday': 'Holiday',
    'good_biking_conditions': 'Good Biking Cond.',
    'temp_squared': 'Temp Squared',
}

# --- Left: SHAP Top 10 bar chart ---
add_text_box(slide, Inches(0.2), Inches(1.4), Inches(7), Inches(0.3),
             "SHAP Global Importance (mean |SHAP value|)", font_size=13, color=PRIMARY, bold=True)

if _shap_data:
    _si = _shap_data['global_importance'][:10]
    _max_shap = _si[0]['mean_abs_shap']
    _max_bar = Inches(3.2)
    for i, item in enumerate(_si):
        top = Inches(1.75 + i * 0.43)
        _sv = item['mean_abs_shap']
        _name = _shap_labels.get(item['feature'], item['feature'])
        _bw = int(_max_bar * (_sv / _max_shap))

        add_text_box(slide, Inches(0.1), top, Inches(2.6), Inches(0.38),
                     _name, font_size=10, color=DARK_TEXT, alignment=PP_ALIGN.RIGHT)
        _bc = ACCENT if i < 3 else SECONDARY
        add_shape(slide, Inches(2.8), top + Inches(0.04), _bw, Inches(0.28),
                  fill_color=_bc, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text_box(slide, Inches(2.8) + _bw + Inches(0.08), top, Inches(0.8), Inches(0.38),
                     f"{_sv:.0f}", font_size=9, color=ACCENT, bold=True)

# --- Right top: Direction analysis ---
add_text_box(slide, Inches(7.2), Inches(1.4), Inches(5.3), Inches(0.3),
             "SHAP Direction (what pushes demand up vs down)", font_size=13, color=PRIMARY, bold=True)

# Demand increasers
add_shape(slide, Inches(7.2), Inches(1.8), Inches(2.5), Inches(0.35),
          fill_color=GREEN, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(7.2), Inches(1.82), Inches(2.5), Inches(0.3),
             "INCREASES Demand", font_size=11, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

up_items = [
    "Temp x Hour (+138) -- warm afternoons",
    "Solar Radiation (+100) -- sunny days",
    "Good Biking Cond. (+19)",
    "Season Summer (+31)",
]
add_simple_bullets(slide, Inches(7.2), Inches(2.2), Inches(2.6), Inches(1.6),
                   up_items, font_size=9)

# Demand decreasers
add_shape(slide, Inches(9.9), Inches(1.8), Inches(2.5), Inches(0.35),
          fill_color=ACCENT, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(9.9), Inches(1.82), Inches(2.5), Inches(0.3),
             "DECREASES Demand", font_size=11, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

down_items = [
    "Humidity (-320) -- strongest factor",
    "Non-Functioning Day (-106)",
    "Rainfall (-86) -- rain deters riders",
    "Wind Speed (-75)",
]
add_simple_bullets(slide, Inches(9.9), Inches(2.2), Inches(2.6), Inches(1.6),
                   down_items, font_size=9)

# --- Right middle: SHAP vs Built-in comparison ---
add_text_box(slide, Inches(7.2), Inches(4.0), Inches(5.3), Inches(0.3),
             "SHAP vs Built-in Feature Importance", font_size=13, color=PRIMARY, bold=True)

comp_box = add_shape(slide, Inches(7.2), Inches(4.35), Inches(5.3), Inches(1.8),
                     fill_color=RGBColor(0xFD, 0xF2, 0xE9), line_color=ACCENT)
comp_box.line.width = Pt(1.5)

add_text_box(slide, Inches(7.4), Inches(4.4), Inches(5), Inches(1.7),
             "Built-in (gain): #1 Functioning Day (12.4%), #2 Temp Category (11.8%)\n"
             "SHAP (marginal):  #1 Humidity (319.7),  #2 Temp x Hour (139.1)\n\n"
             "Why they differ: Built-in counts split frequency (favors\n"
             "one-hot features), while SHAP measures actual prediction\n"
             "impact. Humidity is split less often but each split matters more.",
             font_size=10, color=DARK_TEXT)

# --- Bottom: Key insight + expected value ---
_ev = _shap_data['expected_value'] if _shap_data else 705
add_shape(slide, Inches(0.2), Inches(6.2), Inches(12.4), Inches(0.85),
          fill_color=LIGHT_BLUE, line_color=SECONDARY)
add_text_box(slide, Inches(0.5), Inches(6.25), Inches(8), Inches(0.3),
             "How SHAP Works", font_size=13, color=PRIMARY, bold=True)
add_text_box(slide, Inches(0.5), Inches(6.5), Inches(8), Inches(0.5),
             f"Base prediction = {_ev:.0f} bikes/hr (dataset average). Each feature's SHAP value adds or\n"
             "subtracts from this base. Example: high humidity SHAP = -320 means demand drops by 320 bikes/hr.",
             font_size=10, color=DARK_TEXT)
add_card(slide, Inches(9.2), Inches(6.2), Inches(3.3), Inches(0.85),
         "BASE PREDICTION", f"{_ev:.0f}", "bikes/hr (expected value)",
         value_color=PRIMARY)

add_footer_bar(slide)
add_slide_number(slide, 16, TOTAL_SLIDES)


# ============================================================
# SLIDE 17: WEB APP ARCHITECTURE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Web Application Architecture",
              "Production-ready Flask application with layered modular design")

# --- LAYER 1: Client / Presentation Layer ---
layer1_top = Inches(1.5)
layer1_h = Inches(0.85)
# Layer label
add_shape(slide, Inches(0.4), layer1_top, Inches(1.8), layer1_h,
          fill_color=DARK, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.4), layer1_top + Inches(0.12), Inches(1.8), Inches(0.3),
             "PRESENTATION", font_size=10, color=RGBColor(0xA0, 0xC4, 0xE8),
             bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), layer1_top + Inches(0.4), Inches(1.8), Inches(0.3),
             "Client Layer", font_size=12, color=WHITE, bold=False, alignment=PP_ALIGN.CENTER)

# Client components
client_items = [
    ("HTML / Jinja2", "Templates", RGBColor(0xE8, 0x6F, 0x2C)),
    ("CSS / Style", "Responsive UI", RGBColor(0x27, 0xAE, 0x60)),
    ("JavaScript", "Chart.js / AJAX", RGBColor(0x2E, 0x86, 0xAB)),
    ("Font Awesome", "Icon Library", RGBColor(0x8E, 0x44, 0xAD)),
]
for i, (name, desc, color) in enumerate(client_items):
    cl = Inches(2.5 + i * 2.7)
    add_shape(slide, cl, layer1_top, Inches(2.45), layer1_h,
              fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, cl, layer1_top + Inches(0.12), Inches(2.45), Inches(0.3),
                 name, font_size=13, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, cl, layer1_top + Inches(0.45), Inches(2.45), Inches(0.25),
                 desc, font_size=10, color=RGBColor(0xFF, 0xFF, 0xE0), alignment=PP_ALIGN.CENTER)

# --- Connection arrow area ---
add_shape(slide, Inches(6), Inches(2.4), Inches(1.4), Inches(0.3),
          fill_color=RGBColor(0xDE, 0xE2, 0xE6), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(6), Inches(2.4), Inches(1.4), Inches(0.3),
             "REST API", font_size=9, color=GRAY, bold=True, alignment=PP_ALIGN.CENTER)

# --- LAYER 2: Application / Routing Layer ---
layer2_top = Inches(2.8)
layer2_h = Inches(1.1)

add_shape(slide, Inches(0.4), layer2_top, Inches(1.8), layer2_h,
          fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.4), layer2_top + Inches(0.15), Inches(1.8), Inches(0.3),
             "APPLICATION", font_size=10, color=RGBColor(0xA0, 0xC4, 0xE8),
             bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), layer2_top + Inches(0.42), Inches(1.8), Inches(0.25),
             "Flask Blueprints", font_size=11, color=WHITE, bold=False, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), layer2_top + Inches(0.7), Inches(1.8), Inches(0.25),
             "App Factory", font_size=10, color=RGBColor(0xCC, 0xDD, 0xFF), alignment=PP_ALIGN.CENTER)

# Blueprint boxes
blueprints = [
    ("Pages", "Home, Status\nTest Suite", PRIMARY),
    ("Prediction", "6 Models\nAuto Features", ACCENT),
    ("Analytics", "KPIs, Charts\nSHAP, Radar", SECONDARY),
    ("Export", "CSV, JSON\nReports", GREEN),
    ("Features", "Feature Eng.\nAPI Endpoints", RGBColor(0x8E, 0x44, 0xAD)),
]

for i, (name, desc, color) in enumerate(blueprints):
    bl = Inches(2.5 + i * 2.15)
    add_shape(slide, bl, layer2_top, Inches(1.95), layer2_h,
              fill_color=color, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, bl, layer2_top + Inches(0.1), Inches(1.95), Inches(0.3),
                 name, font_size=13, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, bl, layer2_top + Inches(0.45), Inches(1.95), Inches(0.55),
                 desc, font_size=10, color=RGBColor(0xE8, 0xE8, 0xFF), alignment=PP_ALIGN.CENTER)

# --- Connection arrow area ---
add_shape(slide, Inches(6), Inches(4.0), Inches(1.4), Inches(0.3),
          fill_color=RGBColor(0xDE, 0xE2, 0xE6), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(6), Inches(4.0), Inches(1.4), Inches(0.3),
             "Services", font_size=9, color=GRAY, bold=True, alignment=PP_ALIGN.CENTER)

# --- LAYER 3: Service / Data Layer ---
layer3_top = Inches(4.4)
layer3_h = Inches(0.85)

add_shape(slide, Inches(0.4), layer3_top, Inches(1.8), layer3_h,
          fill_color=RGBColor(0x34, 0x49, 0x5E), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.4), layer3_top + Inches(0.12), Inches(1.8), Inches(0.3),
             "SERVICE", font_size=10, color=RGBColor(0xA0, 0xC4, 0xE8),
             bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), layer3_top + Inches(0.4), Inches(1.8), Inches(0.3),
             "Business Logic", font_size=11, color=WHITE, bold=False, alignment=PP_ALIGN.CENTER)

services = [
    ("Model Loader", "Lazy + Cached"),
    ("Prediction Svc", "Inference Engine"),
    ("Export Service", "Multi-format"),
    ("Email Service", "Notifications"),
    ("History Store", "JSON Persistence"),
]

for i, (name, desc) in enumerate(services):
    sl = Inches(2.5 + i * 2.15)
    add_shape(slide, sl, layer3_top, Inches(1.95), layer3_h,
              fill_color=RGBColor(0x34, 0x49, 0x5E), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, sl, layer3_top + Inches(0.1), Inches(1.95), Inches(0.3),
                 name, font_size=12, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, sl, layer3_top + Inches(0.45), Inches(1.95), Inches(0.25),
                 desc, font_size=10, color=RGBColor(0xBB, 0xCC, 0xDD), alignment=PP_ALIGN.CENTER)

# --- Connection arrow area ---
add_shape(slide, Inches(6), Inches(5.35), Inches(1.4), Inches(0.3),
          fill_color=RGBColor(0xDE, 0xE2, 0xE6), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(6), Inches(5.35), Inches(1.4), Inches(0.3),
             "Models", font_size=9, color=GRAY, bold=True, alignment=PP_ALIGN.CENTER)

# --- LAYER 4: Model / Data Layer ---
layer4_top = Inches(5.75)
layer4_h = Inches(0.7)

add_shape(slide, Inches(0.4), layer4_top, Inches(1.8), layer4_h,
          fill_color=RGBColor(0x1A, 0x1A, 0x2E), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.4), layer4_top + Inches(0.08), Inches(1.8), Inches(0.25),
             "DATA", font_size=10, color=RGBColor(0xA0, 0xC4, 0xE8),
             bold=True, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.4), layer4_top + Inches(0.35), Inches(1.8), Inches(0.25),
             "ML Models + Files", font_size=10, color=WHITE, alignment=PP_ALIGN.CENTER)

models_row = [
    ("Random Forest", "R\u00b2=0.82"),
    ("Improved RF", "R\u00b2=0.88"),
    ("Advanced RF", "R\u00b2=0.92"),
    ("XGBoost", "R\u00b2=0.944"),
    ("LightGBM", "R\u00b2=0.757"),
    ("Stacking", "R\u00b2=0.791"),
]

model_w = Inches(1.7)
for i, (name, score) in enumerate(models_row):
    ml = Inches(2.5 + i * 1.8)
    add_shape(slide, ml, layer4_top, model_w, layer4_h,
              fill_color=RGBColor(0x1A, 0x1A, 0x2E), shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, ml, layer4_top + Inches(0.08), model_w, Inches(0.25),
                 name, font_size=10, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, ml, layer4_top + Inches(0.38), model_w, Inches(0.2),
                 score, font_size=9, color=GREEN, bold=False, alignment=PP_ALIGN.CENTER)

# --- Right side: Tech Stack summary ---
tech_left = Inches(0.4)
tech_top = Inches(6.6)
add_text_box(slide, tech_left, tech_top, Inches(12), Inches(0.3),
             "Tech:  Flask 3.0  |  Jinja2  |  Chart.js  |  scikit-learn  |  XGBoost  |  LightGBM  |  SHAP  |  Gunicorn-ready",
             font_size=10, color=GRAY, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 17, TOTAL_SLIDES)


# ============================================================
# SLIDE 18: WEB APP - PREDICTION INTERFACE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Prediction Interface",
              "End-to-end prediction pipeline from user input to actionable output")

# === PREDICTION FLOW: 3-stage pipeline ===
# Stage labels
stages = [
    ("1", "USER INPUT", "14 Weather & Context\nParameters", Inches(0.5), PRIMARY),
    ("2", "MODEL INFERENCE", "6 ML Models with\nAuto Feature Engineering", Inches(4.6), ACCENT),
    ("3", "PREDICTION OUTPUT", "Bike Count, Confidence\n& Analytics", Inches(8.7), GREEN),
]

for num, title, desc, left, color in stages:
    # Stage number circle
    add_shape(slide, left, Inches(1.55), Inches(0.5), Inches(0.5),
              fill_color=color, shape_type=MSO_SHAPE.OVAL)
    add_text_box(slide, left, Inches(1.58), Inches(0.5), Inches(0.45),
                 num, font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    # Stage title
    add_text_box(slide, left + Inches(0.6), Inches(1.5), Inches(3), Inches(0.35),
                 title, font_size=14, color=color, bold=True)
    # Stage description
    add_text_box(slide, left + Inches(0.6), Inches(1.85), Inches(3.2), Inches(0.5),
                 desc, font_size=11, color=GRAY)

# Flow arrows between stages
for arrow_left in [Inches(4.15), Inches(8.25)]:
    add_shape(slide, arrow_left, Inches(1.65), Inches(0.35), Inches(0.3),
              fill_color=RGBColor(0xDE, 0xE2, 0xE6), shape_type=MSO_SHAPE.RIGHT_ARROW)

# === INPUT PARAMETERS SECTION ===
inp_top = Inches(2.6)
# Section background
add_shape(slide, Inches(0.5), inp_top, Inches(3.8), Inches(4.2),
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6),
          shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
# Section header
add_shape(slide, Inches(0.5), inp_top, Inches(3.8), Inches(0.45),
          fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.7), inp_top + Inches(0.05), Inches(3.4), Inches(0.35),
             "Input Parameters", font_size=13, color=WHITE, bold=True)

# Input parameter groups
param_groups = [
    ("Temporal", "Hour (0-23)"),
    ("Weather", "Temperature, Humidity\nWind Speed, Visibility"),
    ("Atmospheric", "Dew Point, Solar Radiation\nRainfall, Snowfall"),
    ("Context", "Season, Holiday Status\nFunctioning Day"),
    ("Model", "Model Selection\n(RF / XGB / LGBM / Stack)"),
]
for i, (label, detail) in enumerate(param_groups):
    py = inp_top + Inches(0.6 + i * 0.7)
    # Label tag
    add_shape(slide, Inches(0.7), py, Inches(1.15), Inches(0.35),
              fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(0.7), py + Inches(0.02), Inches(1.15), Inches(0.3),
                 label, font_size=10, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    # Detail text
    add_text_box(slide, Inches(2.0), py, Inches(2.2), Inches(0.6),
                 detail, font_size=10, color=DARK_TEXT)

# === MODEL PROCESSING SECTION ===
proc_top = Inches(2.6)
add_shape(slide, Inches(4.6), proc_top, Inches(3.8), Inches(4.2),
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6),
          shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_shape(slide, Inches(4.6), proc_top, Inches(3.8), Inches(0.45),
          fill_color=ACCENT, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(4.8), proc_top + Inches(0.05), Inches(3.4), Inches(0.35),
             "Processing Pipeline", font_size=13, color=WHITE, bold=True)

# Processing steps
proc_steps = [
    ("Input Validation", "Required field check\n& type enforcement"),
    ("Field Normalization", "Korean chars \u2192 ASCII\nModel-specific mapping"),
    ("Feature Engineering", "36 engineered features\nCyclic, interactions, lags"),
    ("Model Prediction", "sklearn / XGBoost / LGBM\nEnsemble inference"),
    ("History Logging", "JSON persistence\nTimestamped records"),
]
for i, (step, detail) in enumerate(proc_steps):
    py = proc_top + Inches(0.6 + i * 0.7)
    # Step box
    add_shape(slide, Inches(4.8), py, Inches(1.5), Inches(0.35),
              fill_color=ACCENT, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, Inches(4.8), py + Inches(0.02), Inches(1.5), Inches(0.3),
                 step, font_size=9, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(6.4), py, Inches(1.9), Inches(0.6),
                 detail, font_size=9, color=DARK_TEXT)

# === OUTPUT SECTION ===
out_top = Inches(2.6)
add_shape(slide, Inches(8.7), out_top, Inches(4.1), Inches(4.2),
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6),
          shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_shape(slide, Inches(8.7), out_top, Inches(4.1), Inches(0.45),
          fill_color=GREEN, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(8.9), out_top + Inches(0.05), Inches(3.7), Inches(0.35),
             "Prediction Output", font_size=13, color=WHITE, bold=True)

# Output items with values
out_items = [
    ("Predicted Bikes", "1,344 bikes/hour", PRIMARY),
    ("Confidence Score", "85% confidence", SECONDARY),
    ("Model Used", "XGBoost (R\u00b2=0.944)", ACCENT),
    ("Feature Summary", "36 features processed", RGBColor(0x8E, 0x44, 0xAD)),
    ("Export Options", "CSV, JSON, Email Report", RGBColor(0x34, 0x49, 0x5E)),
]
for i, (label, value, color) in enumerate(out_items):
    py = out_top + Inches(0.6 + i * 0.7)
    add_shape(slide, Inches(8.9), py, Inches(0.12), Inches(0.35),
              fill_color=color)
    add_text_box(slide, Inches(9.15), py, Inches(1.8), Inches(0.35),
                 label, font_size=10, color=DARK_TEXT, bold=True)
    add_text_box(slide, Inches(10.9), py, Inches(1.7), Inches(0.35),
                 value, font_size=10, color=color, bold=False)

# === API ENDPOINT BAR ===
api_top = Inches(6.35)
add_shape(slide, Inches(0.5), api_top, Inches(12.3), Inches(0.65),
          fill_color=DARK, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, Inches(0.8), api_top + Inches(0.05), Inches(1.2), Inches(0.25),
             "API", font_size=11, color=GREEN, bold=True)
add_text_box(slide, Inches(1.6), api_top + Inches(0.05), Inches(10.5), Inches(0.25),
             'POST /predict?model_type=xgboost&calculate_features=true',
             font_size=12, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")
add_text_box(slide, Inches(1.6), api_top + Inches(0.32), Inches(10.5), Inches(0.25),
             'Response:  { "success": true, "predicted_rented_bikes": 1344, "confidence": 85, "model_type": "xgboost" }',
             font_size=10, color=RGBColor(0x88, 0xBB, 0x88), font_name="Source Code Pro")

add_footer_bar(slide)
add_slide_number(slide, 18, TOTAL_SLIDES)


# ============================================================
# SLIDE 19: WEB APP - ANALYTICS DASHBOARD
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Analytics Dashboard",
              "Real-time monitoring, model comparison, and data-driven insights")

# === KPI CARDS ROW ===
kpis = [
    ("Best Model Accuracy", "94.4%", "XGBoost R\u00b2 Score"),
    ("Engineered Features", "36", "Per advanced model"),
    ("Models Available", "6", "Switchable at runtime"),
    ("Prediction History", "100%", "Persistent logging"),
]

for i, (title, value, sub) in enumerate(kpis):
    add_card(slide, Inches(0.5 + i * 3.1), Inches(1.5), Inches(2.85), Inches(1.15),
             title, value, sub)

# === LEFT COLUMN: Charts & Visualizations ===
col1_left = Inches(0.5)
col1_top = Inches(2.9)
col1_w = Inches(6.0)

add_text_box(slide, col1_left, col1_top, col1_w, Inches(0.35),
             "Charts & Visualizations", font_size=16, color=PRIMARY, bold=True)

# Chart cards in a 2x3 grid
charts = [
    ("Hourly Demand Patterns", "24-hour bar chart from\nprediction history data"),
    ("Model Performance Radar", "5-axis comparison across\nall 6 loaded models"),
    ("Feature Importance", "Top 15 features ranked\nby tree-based importance"),
    ("SHAP Analysis", "Directional feature impact\n(positive / negative effect)"),
    ("KPI Trends", "Accuracy, error, and prediction\ncount with time filtering"),
    ("Data-Driven Insights", "Auto-generated observations\nfrom model state and history"),
]

for i, (name, desc) in enumerate(charts):
    row = i // 2
    col = i % 2
    cx = col1_left + Inches(col * 3.05)
    cy = col1_top + Inches(0.45 + row * 1.15)
    # Card background
    add_shape(slide, cx, cy, Inches(2.85), Inches(1.0),
              fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6),
              shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    # Accent bar
    add_shape(slide, cx, cy, Inches(0.08), Inches(1.0), fill_color=PRIMARY)
    # Chart name
    add_text_box(slide, cx + Inches(0.2), cy + Inches(0.1), Inches(2.5), Inches(0.3),
                 name, font_size=12, color=DARK_TEXT, bold=True)
    # Description
    add_text_box(slide, cx + Inches(0.2), cy + Inches(0.45), Inches(2.5), Inches(0.5),
                 desc, font_size=10, color=GRAY)

# === RIGHT COLUMN: Platform Pages + API Endpoints ===
col2_left = Inches(6.8)
col2_top = Inches(2.9)
col2_w = Inches(6.0)

add_text_box(slide, col2_left, col2_top, col2_w, Inches(0.35),
             "Platform Pages", font_size=16, color=PRIMARY, bold=True)

# Page entries
pages = [
    ("Predictor", "Main prediction interface with model selection and input form"),
    ("Analytics", "Dashboard with KPIs, charts, SHAP analysis, and insights"),
    ("System Status", "Real-time API health, model status, and service monitoring"),
    ("Test Suite", "16 automated tests with actual API validation"),
    ("Feature Engineering", "Interactive feature exploration and calculation APIs"),
]

for i, (page, desc) in enumerate(pages):
    py = col2_top + Inches(0.45 + i * 0.62)
    # Page name tag
    add_shape(slide, col2_left, py, Inches(1.5), Inches(0.35),
              fill_color=PRIMARY, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text_box(slide, col2_left, py + Inches(0.02), Inches(1.5), Inches(0.3),
                 page, font_size=11, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
    # Description
    add_text_box(slide, col2_left + Inches(1.65), py, Inches(4.2), Inches(0.5),
                 desc, font_size=10, color=DARK_TEXT)

# === Analytics API Endpoints row ===
api_top = Inches(6.0)
add_text_box(slide, col2_left, api_top, col2_w, Inches(0.3),
             "Analytics API Endpoints (7 routes)", font_size=12, color=PRIMARY, bold=True)

api_endpoints = [
    "/api/dashboard-metrics",
    "/api/analytics/kpis",
    "/api/analytics/hourly-patterns",
    "/api/analytics/model-performance",
    "/api/analytics/feature-importance",
    "/api/analytics/shap-importance",
    "/api/analytics/insights",
]
# Display as compact row
api_text = "  |  ".join(e.split("/api/")[-1] for e in api_endpoints)
add_shape(slide, col2_left, api_top + Inches(0.3), col2_w, Inches(0.45),
          fill_color=DARK, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE)
add_text_box(slide, col2_left + Inches(0.15), api_top + Inches(0.35), Inches(5.7), Inches(0.35),
             api_text, font_size=9, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")

add_footer_bar(slide)
add_slide_number(slide, 19, TOTAL_SLIDES)


# ============================================================
# SLIDE 20: API DESIGN
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "RESTful API Design  (1/2)",
              "Prediction & Analytics endpoints  -  consistent JSON response format")

# === Prediction + Analytics Endpoints ===
core_endpoints = [
    ("PREDICTION", None, None, None),
    ("POST", "/predict", "Model inference (6 models)", "?model_type=xgboost"),
    ("POST", "/predict/compare", "Multi-model comparison", ""),
    ("GET", "/prediction-history", "Logged prediction records", "?limit=50"),
    ("GET", "/model-info", "Model metadata & features", "?type=advanced"),
    ("ANALYTICS", None, None, None),
    ("GET", "/api/dashboard-metrics", "KPI summary metrics", ""),
    ("GET", "/api/analytics/kpis", "Accuracy, error, count", "?filter=week"),
    ("GET", "/api/analytics/hourly-patterns", "24-hour demand data", "?filter=month"),
    ("GET", "/api/analytics/model-performance", "Radar chart data (6 models)", ""),
    ("GET", "/api/analytics/feature-importance", "Tree-based ranking", "?model=best"),
    ("GET", "/api/analytics/shap-importance", "SHAP directional impact", ""),
    ("GET", "/api/analytics/insights", "Auto-generated insights", ""),
]

# Table header
tbl_left = Inches(0.5)
tbl_w = Inches(12.3)
header_y = Inches(1.5)
add_shape(slide, tbl_left, header_y, tbl_w, Inches(0.42), fill_color=PRIMARY)
add_text_box(slide, tbl_left + Inches(0.1), header_y + Inches(0.04), Inches(0.8), Inches(0.35),
             "Method", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(1.0), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Endpoint", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(4.8), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Description", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(8.5), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Parameters", font_size=11, color=WHITE, bold=True)

row_y = Inches(1.92)
row_h = Inches(0.40)
for method, endpoint, desc, params in core_endpoints:
    if endpoint is None:
        add_shape(slide, tbl_left, row_y, tbl_w, row_h, fill_color=RGBColor(0x34, 0x49, 0x5E))
        add_text_box(slide, tbl_left + Inches(0.1), row_y + Inches(0.04), Inches(4), Inches(0.3),
                     method, font_size=10, color=WHITE, bold=True)
    else:
        bg = LIGHT_BG if int((row_y - Inches(1.92)) / row_h) % 2 == 0 else WHITE
        add_shape(slide, tbl_left, row_y, tbl_w, row_h, fill_color=bg)
        method_color = GREEN if method == "POST" else SECONDARY
        add_text_box(slide, tbl_left + Inches(0.1), row_y + Inches(0.04), Inches(0.8), Inches(0.3),
                     method, font_size=10, color=method_color, bold=True)
        add_text_box(slide, tbl_left + Inches(1.0), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                     endpoint, font_size=10, color=DARK_TEXT, font_name="Source Code Pro")
        add_text_box(slide, tbl_left + Inches(4.8), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                     desc, font_size=10, color=GRAY)
        if params:
            add_text_box(slide, tbl_left + Inches(8.5), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                         params, font_size=9, color=SECONDARY, font_name="Source Code Pro")
    row_y += row_h

# Summary bar
add_text_box(slide, tbl_left, row_y + Inches(0.15), tbl_w, Inches(0.3),
             "All endpoints return JSON  |  Content-Type: application/json  |  Standard HTTP status codes",
             font_size=9, color=GRAY, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 20, TOTAL_SLIDES)


# ============================================================
# SLIDE 21: RESTful API Design (Part 2)
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "RESTful API Design  (2/2)",
              "System, Export & Feature Engineering endpoints")

# === System + Export + Features Endpoints ===
extra_endpoints = [
    ("SYSTEM", None, None, None),
    ("GET", "/api/status", "Health check & model state", ""),
    ("GET", "/metrics", "Performance metrics (R\u00b2, MAE)", ""),
    ("GET", "/api/models", "All loaded models info", ""),
    ("EXPORT", None, None, None),
    ("GET", "/export/predictions/csv", "Download predictions CSV", ""),
    ("GET", "/export/predictions/json", "Download predictions JSON", ""),
    ("GET", "/export/model-report", "Text performance report", ""),
    ("POST", "/export/email", "Email report to recipients", ""),
    ("FEATURES", None, None, None),
    ("GET", "/api/features/advanced", "Feature definitions list", ""),
    ("GET", "/api/features/importance", "Feature ranking data", ""),
    ("POST", "/api/features/calculate", "Compute engineered features", ""),
]

# Table header
tbl_left = Inches(0.5)
tbl_w = Inches(12.3)
header_y = Inches(1.5)
add_shape(slide, tbl_left, header_y, tbl_w, Inches(0.42), fill_color=PRIMARY)
add_text_box(slide, tbl_left + Inches(0.1), header_y + Inches(0.04), Inches(0.8), Inches(0.35),
             "Method", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(1.0), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Endpoint", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(4.8), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Description", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(8.5), header_y + Inches(0.04), Inches(3.5), Inches(0.35),
             "Parameters", font_size=11, color=WHITE, bold=True)

row_y = Inches(1.92)
row_h = Inches(0.42)
for method, endpoint, desc, params in extra_endpoints:
    if endpoint is None:
        add_shape(slide, tbl_left, row_y, tbl_w, row_h, fill_color=RGBColor(0x34, 0x49, 0x5E))
        add_text_box(slide, tbl_left + Inches(0.1), row_y + Inches(0.04), Inches(4), Inches(0.3),
                     method, font_size=10, color=WHITE, bold=True)
    else:
        bg = LIGHT_BG if int((row_y - Inches(1.92)) / row_h) % 2 == 0 else WHITE
        add_shape(slide, tbl_left, row_y, tbl_w, row_h, fill_color=bg)
        method_color = GREEN if method == "POST" else SECONDARY
        add_text_box(slide, tbl_left + Inches(0.1), row_y + Inches(0.04), Inches(0.8), Inches(0.3),
                     method, font_size=10, color=method_color, bold=True)
        add_text_box(slide, tbl_left + Inches(1.0), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                     endpoint, font_size=10, color=DARK_TEXT, font_name="Source Code Pro")
        add_text_box(slide, tbl_left + Inches(4.8), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                     desc, font_size=10, color=GRAY)
        if params:
            add_text_box(slide, tbl_left + Inches(8.5), row_y + Inches(0.04), Inches(3.5), Inches(0.3),
                         params, font_size=9, color=SECONDARY, font_name="Source Code Pro")
    row_y += row_h

# Summary bar
add_text_box(slide, tbl_left, row_y + Inches(0.15), tbl_w, Inches(0.3),
             "Legacy + v2 Blueprint routes  |  25 endpoints total across 5 categories",
             font_size=9, color=GRAY, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 21, TOTAL_SLIDES)


# ============================================================
# SLIDE 22: TESTING
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Testing & Quality Assurance",
              "43 automated tests across 4 modules  -  pytest framework  -  100% pass rate")

# === KPI cards row ===
kpi_data = [
    ("TOTAL TESTS", "43", "All passing"),
    ("UNIT TESTS", "23", "Preprocessing + Features"),
    ("INTEGRATION", "20", "Pages + API endpoints"),
    ("TEST FILES", "4", "Organized by domain"),
]
card_w = Inches(2.8)
card_gap = Inches(0.2)
start_x = Inches(0.5)
for i, (title, value, sub) in enumerate(kpi_data):
    cx = start_x + i * (card_w + card_gap)
    add_card(slide, cx, Inches(1.5), card_w, Inches(1.15),
             title, value, sub, value_color=GREEN)

# === Test module breakdown table ===
tbl_left = Inches(0.5)
tbl_w = Inches(7.3)
header_y = Inches(3.0)
add_shape(slide, tbl_left, header_y, tbl_w, Inches(0.38), fill_color=PRIMARY)
add_text_box(slide, tbl_left + Inches(0.15), header_y + Inches(0.04), Inches(3), Inches(0.3),
             "Test Module", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(3.2), header_y + Inches(0.04), Inches(1), Inches(0.3),
             "Tests", font_size=11, color=WHITE, bold=True)
add_text_box(slide, tbl_left + Inches(4.2), header_y + Inches(0.04), Inches(3), Inches(0.3),
             "Coverage", font_size=11, color=WHITE, bold=True)

modules = [
    ("test_preprocessing.py", "10", "Field normalization, Korean chars, input validation"),
    ("test_feature_engineering.py", "13", "Cyclic encoding, lag features, peak hours, comfort index"),
    ("test_pages.py", "11", "Page routes (5), API status, metrics, model-info"),
    ("test_prediction_api.py", "9", "Predict endpoint (7 cases), prediction history"),
]
row_y = header_y + Inches(0.38)
row_h = Inches(0.38)
for idx, (mod, count, desc) in enumerate(modules):
    bg = LIGHT_BG if idx % 2 == 0 else WHITE
    add_shape(slide, tbl_left, row_y, tbl_w, row_h, fill_color=bg)
    add_text_box(slide, tbl_left + Inches(0.15), row_y + Inches(0.04), Inches(3), Inches(0.3),
                 mod, font_size=10, color=DARK_TEXT, font_name="Source Code Pro")
    add_text_box(slide, tbl_left + Inches(3.2), row_y + Inches(0.04), Inches(1), Inches(0.3),
                 count, font_size=10, color=PRIMARY, bold=True)
    add_text_box(slide, tbl_left + Inches(4.2), row_y + Inches(0.04), Inches(3), Inches(0.3),
                 desc, font_size=9, color=GRAY)
    row_y += row_h

# === Key testing patterns (below table) ===
add_text_box(slide, tbl_left, row_y + Inches(0.2), tbl_w, Inches(0.3),
             "Key Testing Patterns", font_size=13, color=PRIMARY, bold=True)
patterns = [
    "Shared fixtures (conftest.py): app factory, test client, sample inputs (Korean + advanced)",
    "Leakage prevention: lag features verified with shift() to exclude current observation",
    "Multi-model coverage: original, advanced, XGBoost, unified model_type parameter",
    "Graceful degradation: tests accept 503 when models are not loaded",
]
pat_y = row_y + Inches(0.5)
for pat in patterns:
    add_text_box(slide, tbl_left + Inches(0.15), pat_y, Inches(7), Inches(0.28),
                 pat, font_size=9, color=GRAY)
    pat_y += Inches(0.28)

# === Code example panel (right side) ===
code_x = Inches(8.1)
code_w = Inches(4.9)
add_shape(slide, code_x, Inches(3.0), code_w, Inches(4.0),
          fill_color=DARK, line_color=DARK)
add_text_box(slide, code_x + Inches(0.2), Inches(3.1), Inches(4.5), Inches(0.3),
             "conftest.py  -  Shared Fixtures", font_size=10, color=GREEN, bold=True)
add_text_box(slide, code_x + Inches(0.2), Inches(3.45), Inches(4.5), Inches(1.4),
             '@pytest.fixture\n'
             'def client(app):\n'
             '    return app.test_client()\n'
             '\n'
             '@pytest.fixture\n'
             'def sample_prediction_input_korean():\n'
             '    return {\n'
             '        "Hour": 14,\n'
             '        "Temperature(\\uce9c)": 22.5,\n'
             '        ...\n'
             '    }',
             font_size=9, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")

add_text_box(slide, code_x + Inches(0.2), Inches(5.0), Inches(4.5), Inches(0.3),
             "test_prediction_api.py  -  Integration", font_size=10, color=GREEN, bold=True)
add_text_box(slide, code_x + Inches(0.2), Inches(5.35), Inches(4.5), Inches(1.4),
             'def test_predict_returns_200(client, sample):\n'
             '    response = client.post(\n'
             '        "/predict",\n'
             '        data=json.dumps(sample),\n'
             '        content_type="application/json"\n'
             '    )\n'
             '    assert response.status_code in (200, 503)',
             font_size=9, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")

add_footer_bar(slide)
add_slide_number(slide, 22, TOTAL_SLIDES)


# ============================================================
# SLIDE 23: DEPLOYMENT
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Deployment & DevOps",
              "Containerized deployment  -  GitHub Actions CI/CD  -  Pre-commit hooks")

# === Row 1: Three deployment mode cards ===
card_w = Inches(3.85)
card_h = Inches(1.65)
card_y = Inches(1.5)
gap = Inches(0.2)

# Development card
add_shape(slide, Inches(0.5), card_y, card_w, card_h,
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6))
add_text_box(slide, Inches(0.7), card_y + Inches(0.1), Inches(3.5), Inches(0.3),
             "Development", font_size=14, color=PRIMARY, bold=True)
add_text_box(slide, Inches(0.7), card_y + Inches(0.45), Inches(3.5), Inches(1.1),
             "python run.py\n"
             "- Flask dev server + hot reload\n"
             "- Debug mode via FLASK_DEBUG env\n"
             "- Auto browser open on start\n"
             "- .env loaded via python-dotenv",
             font_size=10, color=DARK_TEXT)

# Production card
add_shape(slide, Inches(0.5) + card_w + gap, card_y, card_w, card_h,
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6))
add_text_box(slide, Inches(0.7) + card_w + gap, card_y + Inches(0.1), Inches(3.5), Inches(0.3),
             "Production", font_size=14, color=GREEN, bold=True)
add_text_box(slide, Inches(0.7) + card_w + gap, card_y + Inches(0.45), Inches(3.5), Inches(1.1),
             "gunicorn run:app\n"
             "- 2 WSGI workers, 120s timeout\n"
             "- Module-level app = create_app()\n"
             "- Environment variables for config\n"
             "- Port 5000 (0.0.0.0)",
             font_size=10, color=DARK_TEXT)

# Docker card
add_shape(slide, Inches(0.5) + 2 * (card_w + gap), card_y, card_w, card_h,
          fill_color=LIGHT_BG, line_color=RGBColor(0xDE, 0xE2, 0xE6))
add_text_box(slide, Inches(0.7) + 2 * (card_w + gap), card_y + Inches(0.1), Inches(3.5), Inches(0.3),
             "Docker", font_size=14, color=SECONDARY, bold=True)
add_text_box(slide, Inches(0.7) + 2 * (card_w + gap), card_y + Inches(0.45), Inches(3.5), Inches(1.1),
             "docker compose up\n"
             "- Python 3.10-slim + gcc\n"
             "- Volumes: ./data, ./models mounted\n"
             "- .env file injected via env_file\n"
             "- restart: unless-stopped",
             font_size=10, color=DARK_TEXT)

# === Row 2: CI/CD Pipeline (left) + Pre-commit hooks (right) ===
row2_y = Inches(3.4)

# CI/CD Pipeline panel
ci_w = Inches(7.8)
ci_h = Inches(3.4)
add_shape(slide, Inches(0.5), row2_y, ci_w, ci_h,
          fill_color=DARK, line_color=DARK)
add_text_box(slide, Inches(0.7), row2_y + Inches(0.08), Inches(7), Inches(0.3),
             "GitHub Actions CI  (.github/workflows/ci.yml)", font_size=11, color=GREEN, bold=True)

# Three CI job columns
job_x = Inches(0.7)
job_y = row2_y + Inches(0.45)
job_w = Inches(2.35)
job_h = Inches(2.8)
ci_jobs = [
    ("LINT", [
        "black --check (formatting)",
        "isort --check (imports)",
        "flake8 (style, max 120)",
    ]),
    ("TEST", [
        "pytest tests/ -v",
        "Git LFS checkout",
        "43 tests, --tb=short",
    ]),
    ("SECURITY", [
        "bandit -r app/ (SAST)",
        "safety check (CVEs)",
        "Skip B101 (assert)",
    ]),
]
for i, (name, items) in enumerate(ci_jobs):
    jx = job_x + i * (job_w + Inches(0.1))
    add_shape(slide, jx, job_y, job_w, job_h,
              fill_color=RGBColor(0x2A, 0x2A, 0x3A), line_color=RGBColor(0x3A, 0x3A, 0x4A))
    add_text_box(slide, jx + Inches(0.1), job_y + Inches(0.08), job_w - Inches(0.2), Inches(0.25),
                 name, font_size=10, color=WHITE, bold=True)
    # Trigger info
    add_text_box(slide, jx + Inches(0.1), job_y + Inches(0.35), job_w - Inches(0.2), Inches(0.2),
                 "ubuntu-latest / Python 3.10", font_size=8, color=GRAY)
    item_y = job_y + Inches(0.6)
    for item in items:
        add_text_box(slide, jx + Inches(0.1), item_y, job_w - Inches(0.2), Inches(0.22),
                     item, font_size=9, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")
        item_y += Inches(0.25)

# Trigger note
add_text_box(slide, Inches(0.7), row2_y + ci_h - Inches(0.35), Inches(7), Inches(0.25),
             "Triggers: push/PR to main, master", font_size=9, color=GRAY)

# Pre-commit hooks panel (right side)
hook_x = Inches(8.5)
hook_w = Inches(4.5)
add_shape(slide, hook_x, row2_y, hook_w, ci_h,
          fill_color=DARK, line_color=DARK)
add_text_box(slide, hook_x + Inches(0.2), row2_y + Inches(0.08), Inches(4), Inches(0.3),
             "Pre-commit Hooks", font_size=11, color=GREEN, bold=True)
add_text_box(slide, hook_x + Inches(0.2), row2_y + Inches(0.38), Inches(4), Inches(0.2),
             ".pre-commit-config.yaml", font_size=9, color=GRAY, font_name="Source Code Pro")

hooks = [
    ("black 23.12.1", "Code formatting"),
    ("isort 5.13.2", "Import sorting"),
    ("flake8 7.0.0", "Linting (120 char)"),
    ("trailing-whitespace", "Clean whitespace"),
    ("end-of-file-fixer", "EOF newlines"),
    ("check-yaml", "YAML validation"),
    ("detect-private-key", "Secret detection"),
    ("check-added-large-files", "Max 5 MB limit"),
]
hook_y = row2_y + Inches(0.65)
for name, desc in hooks:
    add_text_box(slide, hook_x + Inches(0.2), hook_y, Inches(2.3), Inches(0.22),
                 name, font_size=9, color=RGBColor(0xA0, 0xE8, 0xA0), font_name="Source Code Pro")
    add_text_box(slide, hook_x + Inches(2.5), hook_y, Inches(1.8), Inches(0.22),
                 desc, font_size=9, color=GRAY)
    hook_y += Inches(0.28)

add_footer_bar(slide)
add_slide_number(slide, 23, TOTAL_SLIDES)


# ============================================================
# SLIDE 24: TECHNOLOGY STACK
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Technology Stack",
              "20+ packages  -  pinned versions in requirements.txt  -  Python 3.10")

# === 6 category columns (2 rows x 3 cols) ===
col_w = Inches(3.9)
col_gap = Inches(0.2)
row1_y = Inches(1.5)
row2_y = Inches(4.0)
col_h = Inches(2.25)

tech_grid = [
    # Row 1
    [
        ("Backend & Web", PRIMARY, [
            ("Flask", "3.0.0", "Web framework"),
            ("Werkzeug", "3.0.1", "WSGI utilities"),
            ("Jinja2", "3.1.2", "Template engine"),
            ("Gunicorn", "21.2.0", "Production WSGI server"),
            ("python-dotenv", "1.0.0", "Environment config"),
        ]),
        ("Data Science", SECONDARY, [
            ("pandas", "2.1.4", "Data manipulation"),
            ("NumPy", "1.26.2", "Numerical computing"),
            ("scikit-learn", "1.3.2", "ML framework (RF)"),
            ("SciPy", "1.11.4", "Scientific computing"),
            ("joblib", "1.3.2", "Model serialization"),
        ]),
        ("ML & Explainability", ACCENT, [
            ("XGBoost", "2.0.3", "Gradient boosting"),
            ("LightGBM", "4.2.0", "Fast gradient boosting"),
            ("SHAP", "0.44.0", "Model explanations"),
            ("Stacking", "-", "Ensemble (RF+XGB+LGBM)"),
        ]),
    ],
    # Row 2
    [
        ("Frontend", RGBColor(0xDB, 0x27, 0x77), [
            ("HTML5 / CSS3", "-", "Semantic markup + styles"),
            ("JavaScript (ES6+)", "-", "Async fetch, DOM"),
            ("Chart.js", "CDN", "Interactive charts"),
            ("Font Awesome", "6.4.0", "Icon library"),
        ]),
        ("Reporting & Export", RGBColor(0xD9, 0x77, 0x06), [
            ("reportlab", "4.0.8", "PDF generation"),
            ("matplotlib", "3.8.2", "Plot visualization"),
            ("openpyxl", "3.1.2", "Excel export"),
            ("Pillow", "10.1.0", "Image processing"),
            ("secure-smtplib", "0.1.1", "Email reports"),
        ]),
        ("DevOps & Testing", GREEN, [
            ("Docker + Compose", "-", "Containerization"),
            ("GitHub Actions", "-", "CI/CD pipeline"),
            ("pytest", "7.4.3", "Test framework"),
            ("black / isort / flake8", "-", "Code quality"),
            ("Git + GitHub", "-", "Version control"),
        ]),
    ],
]

for row_idx, row in enumerate(tech_grid):
    base_y = row1_y if row_idx == 0 else row2_y
    for col_idx, (cat_name, color, items) in enumerate(row):
        cx = Inches(0.5) + col_idx * (col_w + col_gap)
        # Category header
        add_shape(slide, cx, base_y, col_w, Inches(0.38), fill_color=color)
        add_text_box(slide, cx, base_y + Inches(0.04), col_w, Inches(0.3),
                     cat_name, font_size=12, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
        # Items table
        item_y = base_y + Inches(0.42)
        for i, (name, ver, desc) in enumerate(items):
            bg = LIGHT_BG if i % 2 == 0 else WHITE
            add_shape(slide, cx, item_y, col_w, Inches(0.32), fill_color=bg)
            add_text_box(slide, cx + Inches(0.1), item_y + Inches(0.02), Inches(1.4), Inches(0.28),
                         name, font_size=9, color=DARK_TEXT, bold=True)
            add_text_box(slide, cx + Inches(1.5), item_y + Inches(0.02), Inches(0.7), Inches(0.28),
                         ver, font_size=9, color=PRIMARY, font_name="Source Code Pro")
            add_text_box(slide, cx + Inches(2.2), item_y + Inches(0.02), Inches(1.6), Inches(0.28),
                         desc, font_size=8, color=GRAY)
            item_y += Inches(0.32)

add_footer_bar(slide)
add_slide_number(slide, 24, TOTAL_SLIDES)


# ============================================================
# SLIDE 25: CHALLENGES & SOLUTIONS
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Challenges & Solutions",
              "Key technical obstacles encountered during development and how they were resolved")

challenges = [
    ("Data Leakage Risk",
     "Random train/test split leaks future hourly\n"
     "data into training set, inflating metrics",
     "Chronological 80/20 split + TimeSeriesSplit (5-fold) CV.\n"
     "Lag features use shift(1) before rolling to exclude current row."),
    ("Korean Character Encoding",
     "CSV column headers contain Korean char (\uce9c).\n"
     "Windows cp1252 console crashes on print()",
     "UTF-8 TextIOWrapper on stdout/stderr in routes.py.\n"
     "normalize_field_names_for_model() maps between schemas."),
    ("Multi-Model Input Formats",
     "6 models expect different feature schemas:\n"
     "Korean keys (RF) vs lowercase (XGBoost/LightGBM)",
     "Centralized normalization function per model type.\n"
     "calculate_features=true auto-engineers 36 features from 14 raw."),
    ("Large Model Artifacts",
     "Trained .pkl files exceed 100 MB each,\n"
     "cannot be stored in regular Git",
     "Git LFS (.gitattributes: *.pkl filter=lfs).\n"
     "CI uses actions/checkout with lfs: true for test pipeline."),
    ("Model Overfitting",
     "Basic Random Forest: high train score but\n"
     "poor generalization on test data (R\u00b2 = 0.82)",
     "Ensemble methods (Stacking RF+XGB+LGBM), feature engineering\n"
     "(cyclic encoding, lag features, comfort index). Best R\u00b2 = 0.944."),
]

for i, (challenge, problem, solution) in enumerate(challenges):
    top = Inches(1.4 + i * 1.1)

    # Challenge box (orange-tinted)
    add_shape(slide, Inches(0.5), top, Inches(4.8), Inches(0.92),
              fill_color=RGBColor(0xFD, 0xF2, 0xE9), line_color=ACCENT)
    add_text_box(slide, Inches(0.65), top + Inches(0.04), Inches(4.5), Inches(0.28),
                 challenge, font_size=12, color=ACCENT, bold=True)
    add_text_box(slide, Inches(0.65), top + Inches(0.32), Inches(4.5), Inches(0.55),
                 problem, font_size=9, color=DARK_TEXT)

    # Arrow
    add_text_box(slide, Inches(5.4), top + Inches(0.25), Inches(0.5), Inches(0.4),
                 "->", font_size=16, color=GRAY, bold=True)

    # Solution box (green-tinted)
    add_shape(slide, Inches(5.9), top, Inches(6.9), Inches(0.92),
              fill_color=RGBColor(0xE8, 0xF5, 0xE9), line_color=GREEN)
    add_text_box(slide, Inches(6.05), top + Inches(0.04), Inches(6.6), Inches(0.28),
                 "Solution", font_size=12, color=GREEN, bold=True)
    add_text_box(slide, Inches(6.05), top + Inches(0.32), Inches(6.6), Inches(0.55),
                 solution, font_size=9, color=DARK_TEXT)

add_footer_bar(slide)
add_slide_number(slide, 25, TOTAL_SLIDES)


# ============================================================
# SLIDE 26: DEMO SCREENSHOTS PLACEHOLDER
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Live Demo / Screenshots",
              "UrbanFlow web application in action")

# Placeholder boxes for screenshots
placeholders = [
    ("Main Prediction Interface", Inches(0.5), Inches(1.5), Inches(6), Inches(5.2)),
    ("Analytics Dashboard", Inches(6.8), Inches(1.5), Inches(6), Inches(2.4)),
    ("API Status Page", Inches(6.8), Inches(4.2), Inches(6), Inches(2.5)),
]

for title, left, top, width, height in placeholders:
    shape = add_shape(slide, left, top, width, height,
                      fill_color=LIGHT_BG, line_color=RGBColor(0xBB, 0xBB, 0xBB))
    shape.line.width = Pt(2)
    # dash pattern
    add_text_box(slide, left, top + height / 2 - Inches(0.3), width, Inches(0.6),
                 f"[ {title} ]\nInsert screenshot here",
                 font_size=16, color=GRAY, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 26, TOTAL_SLIDES)


# ============================================================
# SLIDE 27: CONCLUSION & FUTURE WORK
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, WHITE)
add_title_bar(slide, "Conclusion & Future Work")

# Achievements
add_text_box(slide, Inches(0.8), Inches(1.5), Inches(6), Inches(0.4),
             "Achievements:", font_size=20, color=PRIMARY, bold=True)
achievements = [
    "Built end-to-end ML pipeline for hourly bike demand prediction",
    "Achieved R2 = 0.944 with XGBoost (94.4% variance explained)",
    "Engineered 36+ features from 13 original columns",
    "Deployed production-ready web app with 5 interactive pages",
    "Ensured model integrity with leakage-free design",
    "Comprehensive testing: 43 tests, all passing",
]
add_simple_bullets(slide, Inches(0.8), Inches(2.0), Inches(6), Inches(3.5),
                   achievements, font_size=15)

# Future work
add_text_box(slide, Inches(7.5), Inches(1.5), Inches(5), Inches(0.4),
             "Future Work:", font_size=20, color=SECONDARY, bold=True)
future = [
    "Real-time weather API integration",
    "Deep learning models (LSTM, Transformer)",
    "Multi-city expansion (beyond Seoul)",
    "Mobile-responsive progressive web app",
    "Automated model retraining pipeline",
    "User authentication & personalization",
]
add_simple_bullets(slide, Inches(7.5), Inches(2.0), Inches(5), Inches(3.5),
                   future, font_size=15)

# Impact statement
impact_box = add_shape(slide, Inches(0.8), Inches(5.6), Inches(11.5), Inches(1),
                       fill_color=PRIMARY)
add_text_box(slide, Inches(1.1), Inches(5.7), Inches(11), Inches(0.8),
             "UrbanFlow demonstrates how machine learning can optimize urban mobility,\n"
             "reducing operational costs and improving bike-sharing service quality.",
             font_size=16, color=WHITE, bold=False, alignment=PP_ALIGN.CENTER)

add_footer_bar(slide)
add_slide_number(slide, 27, TOTAL_SLIDES)


# ============================================================
# SLIDE 28: THANK YOU
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide, PRIMARY)

# Decorative
add_shape(slide, Inches(0), Inches(5.5), SLIDE_W, Inches(2), fill_color=RGBColor(0x0F, 0x2D, 0x53))

add_text_box(slide, Inches(0), Inches(1.5), SLIDE_W, Inches(1),
             "Thank You!", font_size=52, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_shape(slide, Inches(5.5), Inches(2.7), Inches(2.3), Inches(0.06), fill_color=ACCENT)

add_text_box(slide, Inches(0), Inches(3.2), SLIDE_W, Inches(0.6),
             "Questions & Discussion", font_size=28, color=RGBColor(0xA0, 0xC4, 0xE8),
             alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(0), Inches(4.5), SLIDE_W, Inches(0.5),
             "UrbanFlow - Urban Shared Bike Rental Demand Prediction",
             font_size=18, color=WHITE, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(0), Inches(5.8), SLIDE_W, Inches(0.4),
             "Yousra El Harchi | PFE 2025-2026",
             font_size=16, color=RGBColor(0xA0, 0xC4, 0xE8), alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0), Inches(6.3), SLIDE_W, Inches(0.4),
             "github.com/yousraelharchi2004/urban-bike-demand-prediction",
             font_size=14, color=RGBColor(0x80, 0xA0, 0xC0), alignment=PP_ALIGN.CENTER)

add_slide_number(slide, 28, TOTAL_SLIDES)


# ============================================================
# SAVE
# ============================================================
output_path = os.path.join(os.path.dirname(__file__), "Reports",
                           "PFE_Presentation_UrbanFlow.pptx")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {TOTAL_SLIDES}")
