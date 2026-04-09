# Design System: Model Recommendation Studio
**Project ID:** model-prediction-studio

## 1. Visual Theme & Atmosphere
A sophisticated, professional data science interface with warm, earthy tones and subtle gradients. The design evokes trust and precision, using a clean layout with generous whitespace, soft shadows, and a color palette that balances teal accents with warm neutrals. The atmosphere is "analytical yet approachable" - perfect for ML practitioners who need reliable tools without visual clutter.

## 2. Color Palette & Roles
- **Primary Background** (#f4efe7): Warm off-white for main content areas, creating a clean canvas
- **Secondary Background** (#fbf8f2): Slightly warmer alternative for cards and panels
- **Panel Background** (rgba(255, 255, 255, 0.84)): Semi-transparent white for floating elements
- **Text Primary** (#101828): Deep charcoal for headings and important text
- **Text Muted** (#5b6472): Medium gray for secondary information
- **Accent Primary** (#0f766e): Deep teal for actions, highlights, and success states
- **Accent Gold** (#b7791f): Warm gold for warnings and secondary accents
- **Sidebar Background** (#0d1320): Deep navy for the analysis panel
- **Sidebar Panel** (rgba(255, 255, 255, 0.06)): Subtle white overlay for sidebar inputs
- **Sidebar Text** (#f8fafc): Light gray for sidebar content
- **Sidebar Muted** (#cbd5e1): Muted gray for sidebar labels

## 3. Typography Rules
- **Primary Font Family**: "Aptos", "Segoe UI", sans-serif - Clean, modern system font for UI elements
- **Heading Font Family**: "Iowan Old Style", "Palatino Linotype", Georgia, serif - Elegant serif for headings and titles
- **Kicker Font**: "Trebuchet MS", "Verdana" - Bold sans-serif for small caps sections
- **Sizes**: Responsive scaling with clamp() for headings (2.1rem to 3.25rem), standard body text
- **Weights**: 400 for body, 600 for labels, 700 for headings
- **Letter Spacing**: -0.02em for headings, 0.18em for uppercase kickers

## 4. Component Stylings
* **Buttons:** Rounded rectangles (16px radius), 48px minimum height, gradient backgrounds for primary actions, subtle shadows
* **Cards:** Rounded rectangles (22px radius), white backgrounds with subtle shadows, 1px borders
* **Inputs:** Rounded rectangles (16px radius), semi-transparent backgrounds, 1px borders
* **Tabs:** Pill-shaped tabs with gradient active states, smooth transitions
* **Metrics:** Card-style containers with rounded corners, centered content, serif value fonts
* **Sidebar:** Dark gradient background, rounded inputs, glassmorphism effects

## 5. Layout Principles
- **Grid System**: Flexible columns with large gaps (3rem), max-width containers (1380px)
- **Whitespace**: Generous padding (1.6rem top, 3rem bottom), 0.95rem gaps between cards
- **Hierarchy**: Clear visual separation with shadows and backgrounds
- **Responsive**: Desktop-first with clamp() for scalable typography
- **Depth**: Soft, diffused shadows (0 22px 50px rgba(15, 23, 42, 0.09)) for layering