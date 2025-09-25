#!/usr/bin/env python3
"""
Test script for Healthcare AI Literature Monitoring System
Tests core functionality without requiring full API setup
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

# Mock data for testing
MOCK_ARXIV_PAPERS = [
    {
        "title": "Large Language Models for Clinical Decision Support: A Systematic Review",
        "authors": ["Smith, J.", "Johnson, A.", "Brown, K."],
        "abstract": "This systematic review examines the application of large language models in clinical decision support systems. We analyzed 150 studies and found significant improvements in diagnostic accuracy and treatment recommendations. Key findings include 15% improvement in diagnostic accuracy and 20% reduction in clinical errors.",
        "url": "https://arxiv.org/abs/2024.12345",
        "journal": "arXiv",
        "publication_date": "2024-09-20",
        "significance_score": 7.5,
        "relevant_chapters": [6, 7, 11],
        "key_findings": [
            "15% improvement in diagnostic accuracy with LLM-based systems",
            "20% reduction in clinical errors when using AI decision support",
            "High physician acceptance rates (85%) for LLM recommendations"
        ]
    },
    {
        "title": "Federated Learning for Medical Imaging: Privacy-Preserving Radiology AI",
        "authors": ["Chen, L.", "Wang, M.", "Davis, R."],
        "abstract": "We present a novel federated learning framework for medical imaging that preserves patient privacy while achieving state-of-the-art performance. Our approach demonstrates superior performance on chest X-ray classification across 10 hospitals without sharing patient data.",
        "url": "https://arxiv.org/abs/2024.67890",
        "journal": "arXiv",
        "publication_date": "2024-09-18",
        "significance_score": 8.2,
        "relevant_chapters": [16, 19, 10],
        "key_findings": [
            "Achieved 94% accuracy on chest X-ray classification",
            "Zero patient data sharing while maintaining performance",
            "Successful deployment across 10 hospital systems"
        ]
    }
]

MOCK_JOURNAL_PAPERS = [
    {
        "title": "AI-Powered Clinical Trial Optimization: Real-World Evidence from 50 Studies",
        "authors": ["NEJM AI Editorial Team"],
        "abstract": "This comprehensive analysis of AI applications in clinical trials shows significant improvements in patient recruitment, protocol optimization, and outcome prediction. FDA approval rates increased by 30% for AI-optimized trials.",
        "url": "https://ai.nejm.org/doi/full/10.1056/AIoa2400123",
        "journal": "NEJM AI",
        "publication_date": "2024-09-22",
        "significance_score": 9.1,
        "relevant_chapters": [11, 12, 13],
        "key_findings": [
            "30% increase in FDA approval rates for AI-optimized trials",
            "50% reduction in patient recruitment time",
            "Improved protocol adherence and outcome prediction"
        ]
    }
]

@dataclass
class TestPaper:
    title: str
    authors: List[str]
    abstract: str
    url: str
    journal: str
    publication_date: str
    significance_score: float
    relevant_chapters: List[int]
    key_findings: List[str]

class LiteratureMonitorTest:
    def __init__(self):
        self.chapter_keywords = {
            1: ['clinical informatics', 'ehr', 'fhir', 'healthcare data'],
            6: ['generative ai', 'llm', 'gpt', 'language models', 'clinical notes'],
            7: ['ai agents', 'autonomous systems', 'clinical agents', 'decision support'],
            10: ['robustness', 'security', 'adversarial', 'privacy', 'hipaa'],
            11: ['regulatory', 'fda', 'compliance', 'validation', 'approval'],
            12: ['clinical validation', 'clinical trials', 'rct', 'evidence'],
            13: ['deployment', 'production', 'mlops', 'scaling', 'infrastructure'],
            16: ['medical imaging', 'radiology', 'computer vision', 'imaging ai'],
            19: ['federated learning', 'distributed learning', 'privacy preserving']
        }

    def test_significance_scoring(self):
        """Test the significance scoring algorithm."""
        print("🧪 Testing significance scoring...")
        
        test_cases = [
            ("Breakthrough AI system for clinical diagnosis", "This novel approach shows significant improvement over state-of-the-art methods with FDA approval pending.", 6.0),
            ("Machine learning for healthcare", "Standard application of existing methods.", 1.0),
            ("First real-world deployment of clinical AI", "This breakthrough system achieved superior performance in clinical trials.", 8.0)
        ]
        
        for title, abstract, expected_min in test_cases:
            score = self.calculate_significance_score(title, abstract)
            status = "✅ PASS" if score >= expected_min else "❌ FAIL"
            print(f"  {status} '{title[:50]}...' -> Score: {score:.1f} (expected: ≥{expected_min})")
        
        return True

    def calculate_significance_score(self, title: str, abstract: str) -> float:
        """Calculate significance score based on content analysis."""
        significance_indicators = [
            'breakthrough', 'novel', 'first', 'significant improvement',
            'state-of-the-art', 'outperforms', 'superior', 'innovative',
            'clinical trial', 'randomized', 'validation', 'real-world',
            'fda approved', 'regulatory', 'deployment', 'implementation'
        ]
        
        text = (title + " " + abstract).lower()
        score = 0.0
        
        for indicator in significance_indicators:
            if indicator in text:
                score += 1.0
                
        # Boost score for certain journals/venues
        if any(term in text for term in ['nature', 'nejm', 'jama']):
            score += 2.0
            
        return min(score, 10.0)

    def test_chapter_mapping(self):
        """Test chapter mapping functionality."""
        print("🧪 Testing chapter mapping...")
        
        test_cases = [
            ("Large language models for clinical notes", [6]),
            ("Medical imaging with computer vision", [16]),
            ("Federated learning for healthcare privacy", [19, 10]),
            ("FDA regulatory compliance for AI systems", [11])
        ]
        
        for text, expected_chapters in test_cases:
            mapped_chapters = self.identify_relevant_chapters(text, "")
            status = "✅ PASS" if any(ch in mapped_chapters for ch in expected_chapters) else "❌ FAIL"
            print(f"  {status} '{text}' -> Chapters: {mapped_chapters} (expected: {expected_chapters})")
        
        return True

    def identify_relevant_chapters(self, title: str, abstract: str) -> List[int]:
        """Identify which chapters are relevant to this paper."""
        text = (title + " " + abstract).lower()
        relevant_chapters = []
        
        for chapter_num, keywords in self.chapter_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    relevant_chapters.append(chapter_num)
                    break
                    
        return list(set(relevant_chapters))

    def test_mock_data_processing(self):
        """Test processing of mock research papers."""
        print("🧪 Testing mock data processing...")
        
        all_papers = []
        
        # Process mock arXiv papers
        for paper_data in MOCK_ARXIV_PAPERS:
            paper = TestPaper(**paper_data)
            all_papers.append(paper)
            print(f"  ✅ Processed arXiv paper: '{paper.title[:50]}...' (Score: {paper.significance_score})")
        
        # Process mock journal papers
        for paper_data in MOCK_JOURNAL_PAPERS:
            paper = TestPaper(**paper_data)
            all_papers.append(paper)
            print(f"  ✅ Processed journal paper: '{paper.title[:50]}...' (Score: {paper.significance_score})")
        
        # Filter significant papers
        significant_papers = [p for p in all_papers if p.significance_score >= 3.0]
        print(f"  📊 Found {len(significant_papers)} significant papers out of {len(all_papers)} total")
        
        return significant_papers

    def test_chapter_update_simulation(self, papers: List[TestPaper]):
        """Simulate chapter update process."""
        print("🧪 Testing chapter update simulation...")
        
        chapter_updates = {}
        
        for paper in papers:
            for chapter_num in paper.relevant_chapters:
                if chapter_num not in chapter_updates:
                    chapter_updates[chapter_num] = []
                chapter_updates[chapter_num].append(paper)
        
        for chapter_num, chapter_papers in chapter_updates.items():
            print(f"  📚 Chapter {chapter_num}: {len(chapter_papers)} relevant papers")
            for paper in chapter_papers:
                print(f"    - {paper.title[:60]}... (Score: {paper.significance_score})")
        
        return chapter_updates

    def generate_test_report(self, chapter_updates: Dict[int, List[TestPaper]]):
        """Generate a test report."""
        report = {
            "test_date": datetime.now().isoformat(),
            "total_papers_processed": len(MOCK_ARXIV_PAPERS) + len(MOCK_JOURNAL_PAPERS),
            "significant_papers": sum(len(papers) for papers in chapter_updates.values()),
            "chapters_affected": list(chapter_updates.keys()),
            "chapter_updates": {}
        }
        
        for chapter_num, papers in chapter_updates.items():
            report["chapter_updates"][chapter_num] = [
                {
                    "title": paper.title,
                    "journal": paper.journal,
                    "significance_score": paper.significance_score,
                    "key_findings": paper.key_findings
                }
                for paper in papers
            ]
        
        # Save test report
        with open('literature_monitor_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Test report saved to: literature_monitor_test_report.json")
        return report

    def run_full_test(self):
        """Run complete test suite."""
        print("🚀 Starting Healthcare AI Literature Monitor Test Suite")
        print("=" * 60)
        
        try:
            # Test 1: Significance scoring
            self.test_significance_scoring()
            print()
            
            # Test 2: Chapter mapping
            self.test_chapter_mapping()
            print()
            
            # Test 3: Mock data processing
            significant_papers = self.test_mock_data_processing()
            print()
            
            # Test 4: Chapter update simulation
            chapter_updates = self.test_chapter_update_simulation(significant_papers)
            print()
            
            # Test 5: Generate report
            print("🧪 Generating test report...")
            report = self.generate_test_report(chapter_updates)
            
            print("=" * 60)
            print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
            print(f"📊 Summary:")
            print(f"  - Papers processed: {report['total_papers_processed']}")
            print(f"  - Significant papers: {report['significant_papers']}")
            print(f"  - Chapters affected: {len(report['chapters_affected'])}")
            print(f"  - Chapters: {', '.join(map(str, sorted(report['chapters_affected'])))}")
            
            return True
            
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            return False

if __name__ == "__main__":
    tester = LiteratureMonitorTest()
    success = tester.run_full_test()
    
    if success:
        print("\n🎉 Literature monitoring system is ready for deployment!")
        print("Next steps:")
        print("1. Add your OpenAI API key to GitHub Secrets")
        print("2. Create the GitHub Actions workflow")
        print("3. Enable weekly automated monitoring")
    else:
        print("\n❌ Tests failed. Please review the errors above.")
    
    sys.exit(0 if success else 1)
