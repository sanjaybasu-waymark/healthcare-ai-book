---
layout: default
title: "Navigation Test"
---

# Navigation Link Test Page

This page tests all chapter navigation links to ensure they work correctly.

## Chapter Links Test

{% for chapter in site.chapters %}
- [{{ chapter.title }}]({{ chapter.url }}) - {{ chapter.title }}
{% endfor %}

## Manual Link Test

### Part I: Foundations
- [Chapter 1: Clinical Informatics](chapters/01-clinical-informatics/)
- [Chapter 2: Mathematical Foundations](chapters/02-mathematical-foundations/)
- [Chapter 3: Healthcare Data Engineering](chapters/03-healthcare-data-engineering/)
- [Chapter 4: Structured Machine Learning](chapters/04-structured-ml-clinical/)
- [Chapter 5: Reinforcement Learning](chapters/05-reinforcement-learning-healthcare/)
- [Chapter 6: Generative AI](chapters/06-generative-ai-healthcare/)
- [Chapter 7: AI Agents](chapters/07-ai-agents-healthcare/)

### Part II: Trustworthy AI
- [Chapter 8: Bias Detection](chapters/08-bias-detection-mitigation/)
- [Chapter 9: Interpretability](chapters/09-interpretability-explainability/)
- [Chapter 10: Robustness](chapters/10-robustness-security/)
- [Chapter 11: Regulatory Compliance](chapters/11-regulatory-compliance/)
- [Chapter 12: Clinical Validation](chapters/12-clinical-validation-frameworks/)

### Part III: Deployment
- [Chapter 13: Deployment Strategies](chapters/13-real-world-deployment-strategies/)
- [Chapter 14: Population Health](chapters/14-population-health-ai-systems/)
- [Chapter 15: Health Equity](chapters/15-health-equity-applications/)
- [Chapter 16: Medical Imaging](chapters/16-advanced-medical-imaging-ai/)
- [Chapter 17: Clinical NLP](chapters/17-clinical-nlp-at-scale/)
- [Chapter 18: Multimodal AI](chapters/18-multimodal-ai-systems/)
- [Chapter 19: Federated Learning](chapters/19-federated-learning-healthcare/)

### Part IV: Specialized Applications
- [Chapter 20: Edge Computing](chapters/20-edge-computing-healthcare/)
- [Chapter 21: AI-Assisted Surgery](chapters/21-ai-assisted-surgery-and-robotic-applications/)
- [Chapter 22: Drug Discovery](chapters/22-drug-discovery-and-development-with-ai/)
- [Chapter 23: Precision Medicine](chapters/23-precision-medicine-and-personalized-healthcare/)
- [Chapter 24: Healthcare Operations](chapters/24-healthcare-operations-and-resource-optimization/)
- [Chapter 25: Quality Improvement](chapters/25-quality-improvement-and-patient-safety/)

### Part V: Future Directions
- [Chapter 26: Emerging Technologies](chapters/26-emerging-technologies-and-future-directions/)
- [Chapter 27: Case Studies](chapters/27-case-studies-and-real-world-applications/)
- [Chapter 28: Causal Inference](chapters/28-causal-inference-in-healthcare-ai/)
- [Chapter 29: Environmental Sustainability](chapters/29-environmental-sustainability-in-healthcare-ai/)

If all links above work correctly, navigation is fixed!
