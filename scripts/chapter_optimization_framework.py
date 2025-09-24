#!/usr/bin/env python3
"""
Comprehensive Chapter Optimization Framework
Ensures expert-level quality, accurate citations, and complete validation

This framework systematically optimizes every chapter for:
- Writing quality and coherence
- Expert review standards
- Code review and validation
- Citation accuracy and completeness
- Bibliography completeness

Author: Sanjay Basu, MD PhD (Waymark)
"""

import os
import re
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChapterQualityMetrics:
    """Quality metrics for chapter assessment"""
    chapter_id: str
    writing_quality_score: float
    code_quality_score: float
    citation_completeness_score: float
    expert_review_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]

@dataclass
class CodeBlock:
    """Represents a code block in a chapter"""
    language: str
    content: str
    line_start: int
    line_end: int
    has_attribution: bool
    has_tests: bool
    has_documentation: bool

@dataclass
class Citation:
    """Represents a citation in a chapter"""
    citation_key: str
    line_number: int
    context: str
    is_valid: bool
    bibliography_entry: Optional[str]

class ChapterOptimizationFramework:
    """
    Comprehensive framework for optimizing healthcare AI book chapters
    """
    
    def __init__(self, book_root: str):
        self.book_root = book_root
        self.chapters_dir = os.path.join(book_root, "_chapters")
        self.bibliography_file = os.path.join(book_root, "_bibliography", "references.bib")
        self.quality_standards = self._load_quality_standards()
        self.citation_database = self._load_citation_database()
        
    def _load_quality_standards(self) -> Dict[str, Any]:
        """Load quality standards for chapter optimization"""
        return {
            'writing_quality': {
                'min_words_per_chapter': 3000,
                'max_words_per_chapter': 8000,
                'min_sections_per_chapter': 5,
                'readability_target': 'graduate_level',
                'required_elements': [
                    'learning_objectives',
                    'chapter_overview', 
                    'key_takeaways',
                    'bibliography',
                    'next_steps'
                ]
            },
            'code_quality': {
                'min_code_blocks_per_chapter': 3,
                'required_code_elements': [
                    'docstrings',
                    'type_hints',
                    'error_handling',
                    'logging',
                    'tests'
                ],
                'max_lines_per_function': 50,
                'min_test_coverage': 0.8
            },
            'citation_standards': {
                'min_citations_per_chapter': 10,
                'required_citation_types': [
                    'journal_articles',
                    'conference_papers',
                    'books',
                    'software'
                ],
                'max_self_citations': 0.2,
                'require_doi_when_available': True
            },
            'expert_review': {
                'technical_accuracy': 'high',
                'clinical_relevance': 'high',
                'implementation_quality': 'production_ready',
                'bias_awareness': 'comprehensive'
            }
        }
    
    def _load_citation_database(self) -> Dict[str, Any]:
        """Load and parse bibliography database"""
        if not os.path.exists(self.bibliography_file):
            logger.warning(f"Bibliography file not found: {self.bibliography_file}")
            return {}
        
        try:
            with open(self.bibliography_file, 'r', encoding='utf-8') as f:
                bib_content = f.read()
            
            # Parse BibTeX entries (simplified parser)
            citations = {}
            entries = re.findall(r'@(\w+)\{([^,]+),([^}]+)\}', bib_content, re.DOTALL)
            
            for entry_type, key, content in entries:
                citations[key] = {
                    'type': entry_type,
                    'key': key,
                    'content': content.strip()
                }
            
            return citations
        except Exception as e:
            logger.error(f"Error loading bibliography: {e}")
            return {}
    
    def optimize_chapter(self, chapter_file: str) -> ChapterQualityMetrics:
        """
        Comprehensively optimize a single chapter
        
        Args:
            chapter_file: Path to chapter markdown file
            
        Returns:
            Quality metrics and optimization results
        """
        logger.info(f"Optimizing chapter: {chapter_file}")
        
        # Read chapter content
        with open(chapter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract chapter metadata
        chapter_id = os.path.basename(chapter_file).replace('.md', '')
        
        # Perform comprehensive analysis
        writing_metrics = self._analyze_writing_quality(content)
        code_metrics = self._analyze_code_quality(content)
        citation_metrics = self._analyze_citations(content)
        expert_metrics = self._analyze_expert_standards(content)
        
        # Calculate overall score
        overall_score = (
            writing_metrics['score'] * 0.3 +
            code_metrics['score'] * 0.3 +
            citation_metrics['score'] * 0.2 +
            expert_metrics['score'] * 0.2
        )
        
        # Compile issues and recommendations
        issues = []
        recommendations = []
        
        issues.extend(writing_metrics.get('issues', []))
        issues.extend(code_metrics.get('issues', []))
        issues.extend(citation_metrics.get('issues', []))
        issues.extend(expert_metrics.get('issues', []))
        
        recommendations.extend(writing_metrics.get('recommendations', []))
        recommendations.extend(code_metrics.get('recommendations', []))
        recommendations.extend(citation_metrics.get('recommendations', []))
        recommendations.extend(expert_metrics.get('recommendations', []))
        
        return ChapterQualityMetrics(
            chapter_id=chapter_id,
            writing_quality_score=writing_metrics['score'],
            code_quality_score=code_metrics['score'],
            citation_completeness_score=citation_metrics['score'],
            expert_review_score=expert_metrics['score'],
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _analyze_writing_quality(self, content: str) -> Dict[str, Any]:
        """Analyze writing quality and coherence"""
        lines = content.split('\n')
        word_count = len(content.split())
        
        # Check for required elements
        required_elements = self.quality_standards['writing_quality']['required_elements']
        missing_elements = []
        
        for element in required_elements:
            if element.replace('_', ' ').lower() not in content.lower():
                missing_elements.append(element)
        
        # Analyze structure
        headings = [line for line in lines if line.startswith('#')]
        sections = len([h for h in headings if h.startswith('## ')])
        
        # Calculate score
        score = 1.0
        issues = []
        recommendations = []
        
        # Word count check
        min_words = self.quality_standards['writing_quality']['min_words_per_chapter']
        max_words = self.quality_standards['writing_quality']['max_words_per_chapter']
        
        if word_count < min_words:
            score -= 0.2
            issues.append(f"Chapter too short: {word_count} words (minimum: {min_words})")
            recommendations.append("Expand content with more detailed explanations and examples")
        elif word_count > max_words:
            score -= 0.1
            issues.append(f"Chapter too long: {word_count} words (maximum: {max_words})")
            recommendations.append("Consider splitting into multiple chapters or condensing content")
        
        # Section structure check
        min_sections = self.quality_standards['writing_quality']['min_sections_per_chapter']
        if sections < min_sections:
            score -= 0.2
            issues.append(f"Insufficient sections: {sections} (minimum: {min_sections})")
            recommendations.append("Add more sections to improve content organization")
        
        # Missing elements check
        if missing_elements:
            score -= 0.1 * len(missing_elements)
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
            recommendations.append("Add all required chapter elements")
        
        return {
            'score': max(0.0, score),
            'word_count': word_count,
            'sections': sections,
            'missing_elements': missing_elements,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _analyze_code_quality(self, content: str) -> Dict[str, Any]:
        """Analyze code quality and completeness"""
        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        
        score = 1.0
        issues = []
        recommendations = []
        
        # Check minimum code blocks
        min_blocks = self.quality_standards['code_quality']['min_code_blocks_per_chapter']
        if len(code_blocks) < min_blocks:
            score -= 0.3
            issues.append(f"Insufficient code blocks: {len(code_blocks)} (minimum: {min_blocks})")
            recommendations.append("Add more practical code examples")
        
        # Analyze each code block
        for i, block in enumerate(code_blocks):
            block_issues = self._validate_code_block(block)
            if block_issues:
                score -= 0.1
                issues.extend([f"Code block {i+1}: {issue}" for issue in block_issues])
        
        # Check for attribution
        attribution_blocks = content.count('{% include attribution.html')
        if attribution_blocks < len([b for b in code_blocks if len(b.content) > 100]):
            score -= 0.2
            issues.append("Missing attribution for substantial code implementations")
            recommendations.append("Add attribution blocks for all substantial code examples")
        
        return {
            'score': max(0.0, score),
            'code_blocks': len(code_blocks),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract and analyze code blocks from content"""
        code_blocks = []
        lines = content.split('\n')
        
        in_code_block = False
        current_block = []
        current_language = ''
        start_line = 0
        
        for i, line in enumerate(lines):
            if line.startswith('```'):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    current_language = line[3:].strip()
                    start_line = i + 1
                    current_block = []
                else:
                    # End of code block
                    in_code_block = False
                    if current_block:
                        block_content = '\n'.join(current_block)
                        code_blocks.append(CodeBlock(
                            language=current_language,
                            content=block_content,
                            line_start=start_line,
                            line_end=i,
                            has_attribution=False,  # Will be checked separately
                            has_tests=self._has_tests(block_content),
                            has_documentation=self._has_documentation(block_content)
                        ))
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def _validate_code_block(self, block: CodeBlock) -> List[str]:
        """Validate individual code block quality"""
        issues = []
        
        if block.language == 'python':
            # Check for docstrings
            if '"""' not in block.content and "'''" not in block.content:
                issues.append("Missing docstring")
            
            # Check for type hints
            if 'def ' in block.content and '->' not in block.content:
                issues.append("Missing type hints")
            
            # Check for error handling
            if 'try:' not in block.content and 'except' not in block.content:
                if len(block.content.split('\n')) > 20:  # Only for substantial code
                    issues.append("Missing error handling")
            
            # Check function length
            functions = re.findall(r'def\s+\w+.*?(?=\ndef|\Z)', block.content, re.DOTALL)
            for func in functions:
                if len(func.split('\n')) > self.quality_standards['code_quality']['max_lines_per_function']:
                    issues.append("Function too long")
        
        return issues
    
    def _has_tests(self, code: str) -> bool:
        """Check if code includes tests"""
        test_indicators = ['def test_', 'assert ', 'unittest', 'pytest']
        return any(indicator in code for indicator in test_indicators)
    
    def _has_documentation(self, code: str) -> bool:
        """Check if code includes documentation"""
        doc_indicators = ['"""', "'''", '# ', 'Args:', 'Returns:']
        return any(indicator in code for indicator in doc_indicators)
    
    def _analyze_citations(self, content: str) -> Dict[str, Any]:
        """Analyze citation completeness and accuracy"""
        # Extract citations
        citations = self._extract_citations(content)
        
        score = 1.0
        issues = []
        recommendations = []
        
        # Check minimum citations
        min_citations = self.quality_standards['citation_standards']['min_citations_per_chapter']
        if len(citations) < min_citations:
            score -= 0.4
            issues.append(f"Insufficient citations: {len(citations)} (minimum: {min_citations})")
            recommendations.append("Add more citations to support claims and methodologies")
        
        # Validate each citation
        invalid_citations = []
        for citation in citations:
            if not self._validate_citation(citation):
                invalid_citations.append(citation.citation_key)
        
        if invalid_citations:
            score -= 0.2
            issues.append(f"Invalid citations: {', '.join(invalid_citations)}")
            recommendations.append("Fix invalid citations and ensure bibliography entries exist")
        
        # Check citation diversity
        citation_types = set()
        for citation in citations:
            if citation.citation_key in self.citation_database:
                citation_types.add(self.citation_database[citation.citation_key]['type'])
        
        required_types = self.quality_standards['citation_standards']['required_citation_types']
        missing_types = [t for t in required_types if t not in citation_types]
        
        if missing_types:
            score -= 0.1
            issues.append(f"Missing citation types: {', '.join(missing_types)}")
            recommendations.append("Include diverse citation types (journals, conferences, books)")
        
        return {
            'score': max(0.0, score),
            'citations': len(citations),
            'invalid_citations': invalid_citations,
            'citation_types': list(citation_types),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _extract_citations(self, content: str) -> List[Citation]:
        """Extract citations from content"""
        citations = []
        lines = content.split('\n')
        
        # Look for {% cite key %} patterns
        cite_pattern = r'\{% cite ([^%]+) %\}'
        
        for i, line in enumerate(lines):
            matches = re.findall(cite_pattern, line)
            for match in matches:
                citation_key = match.strip()
                citations.append(Citation(
                    citation_key=citation_key,
                    line_number=i + 1,
                    context=line.strip(),
                    is_valid=citation_key in self.citation_database,
                    bibliography_entry=self.citation_database.get(citation_key)
                ))
        
        return citations
    
    def _validate_citation(self, citation: Citation) -> bool:
        """Validate individual citation"""
        return citation.citation_key in self.citation_database
    
    def _analyze_expert_standards(self, content: str) -> Dict[str, Any]:
        """Analyze adherence to expert review standards"""
        score = 1.0
        issues = []
        recommendations = []
        
        # Check for bias awareness
        bias_keywords = ['bias', 'fairness', 'equity', 'demographic', 'disparity']
        if not any(keyword in content.lower() for keyword in bias_keywords):
            score -= 0.2
            issues.append("Missing bias awareness discussion")
            recommendations.append("Add discussion of bias and fairness considerations")
        
        # Check for clinical validation
        validation_keywords = ['validation', 'clinical trial', 'rct', 'evaluation']
        if not any(keyword in content.lower() for keyword in validation_keywords):
            score -= 0.2
            issues.append("Missing clinical validation discussion")
            recommendations.append("Add clinical validation methodology")
        
        # Check for regulatory considerations
        regulatory_keywords = ['fda', 'hipaa', 'regulation', 'compliance']
        if not any(keyword in content.lower() for keyword in regulatory_keywords):
            score -= 0.1
            issues.append("Missing regulatory considerations")
            recommendations.append("Add regulatory compliance discussion")
        
        # Check for population health focus
        population_keywords = ['population health', 'public health', 'health equity']
        if not any(keyword in content.lower() for keyword in population_keywords):
            score -= 0.1
            issues.append("Missing population health perspective")
            recommendations.append("Add population health applications")
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def optimize_all_chapters(self) -> Dict[str, ChapterQualityMetrics]:
        """Optimize all chapters in the book"""
        results = {}
        
        if not os.path.exists(self.chapters_dir):
            logger.error(f"Chapters directory not found: {self.chapters_dir}")
            return results
        
        # Find all chapter files
        chapter_files = []
        for file in os.listdir(self.chapters_dir):
            if file.endswith('.md'):
                chapter_files.append(os.path.join(self.chapters_dir, file))
        
        chapter_files.sort()
        
        # Optimize each chapter
        for chapter_file in chapter_files:
            try:
                metrics = self.optimize_chapter(chapter_file)
                results[metrics.chapter_id] = metrics
                logger.info(f"Optimized {metrics.chapter_id}: Score {metrics.overall_score:.2f}")
            except Exception as e:
                logger.error(f"Error optimizing {chapter_file}: {e}")
        
        return results
    
    def generate_optimization_report(self, results: Dict[str, ChapterQualityMetrics]) -> str:
        """Generate comprehensive optimization report"""
        report = f"""
HEALTHCARE AI BOOK OPTIMIZATION REPORT
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SUMMARY
---------------
Total Chapters Analyzed: {len(results)}
Average Overall Score: {sum(r.overall_score for r in results.values()) / len(results):.2f}
Average Writing Quality: {sum(r.writing_quality_score for r in results.values()) / len(results):.2f}
Average Code Quality: {sum(r.code_quality_score for r in results.values()) / len(results):.2f}
Average Citation Score: {sum(r.citation_completeness_score for r in results.values()) / len(results):.2f}
Average Expert Review Score: {sum(r.expert_review_score for r in results.values()) / len(results):.2f}

CHAPTER-BY-CHAPTER ANALYSIS
---------------------------
"""
        
        for chapter_id, metrics in sorted(results.items()):
            report += f"""
{chapter_id.upper()}
Overall Score: {metrics.overall_score:.2f}
- Writing Quality: {metrics.writing_quality_score:.2f}
- Code Quality: {metrics.code_quality_score:.2f}
- Citation Completeness: {metrics.citation_completeness_score:.2f}
- Expert Review: {metrics.expert_review_score:.2f}

Issues ({len(metrics.issues)}):
{chr(10).join(f"  - {issue}" for issue in metrics.issues)}

Recommendations ({len(metrics.recommendations)}):
{chr(10).join(f"  - {rec}" for rec in metrics.recommendations)}
"""
        
        # Priority recommendations
        all_issues = []
        for metrics in results.values():
            all_issues.extend([(metrics.chapter_id, issue) for issue in metrics.issues])
        
        report += f"""
PRIORITY ACTIONS REQUIRED
------------------------
Total Issues Identified: {len(all_issues)}

High Priority Issues:
"""
        
        # Categorize issues by priority
        high_priority = [issue for chapter, issue in all_issues if any(keyword in issue.lower() 
                        for keyword in ['insufficient', 'missing', 'invalid'])]
        
        for issue in high_priority[:10]:  # Top 10 high priority
            report += f"  - {issue}\n"
        
        return report
    
    def save_optimization_results(self, results: Dict[str, ChapterQualityMetrics], 
                                output_file: str):
        """Save optimization results to file"""
        # Convert to serializable format
        serializable_results = {}
        for chapter_id, metrics in results.items():
            serializable_results[chapter_id] = {
                'chapter_id': metrics.chapter_id,
                'writing_quality_score': metrics.writing_quality_score,
                'code_quality_score': metrics.code_quality_score,
                'citation_completeness_score': metrics.citation_completeness_score,
                'expert_review_score': metrics.expert_review_score,
                'overall_score': metrics.overall_score,
                'issues': metrics.issues,
                'recommendations': metrics.recommendations,
                'timestamp': datetime.now().isoformat()
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Optimization results saved to: {output_file}")

def main():
    """Main optimization function"""
    book_root = "/home/ubuntu/healthcare-ai-book-optimized"
    
    # Initialize optimization framework
    optimizer = ChapterOptimizationFramework(book_root)
    
    # Optimize all chapters
    print("Starting comprehensive chapter optimization...")
    results = optimizer.optimize_all_chapters()
    
    # Generate report
    report = optimizer.generate_optimization_report(results)
    print(report)
    
    # Save results
    results_file = os.path.join(book_root, "optimization_results.json")
    optimizer.save_optimization_results(results, results_file)
    
    # Save report
    report_file = os.path.join(book_root, "optimization_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nOptimization complete!")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")

if __name__ == "__main__":
    main()
