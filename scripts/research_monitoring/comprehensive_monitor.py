#!/usr/bin/env python3
"""
Comprehensive Research Monitoring System for Healthcare AI Book
Monitors multiple sources for new research and updates book content automatically

This is an original educational implementation demonstrating automated research
monitoring and content update systems for technical publications.

Author: Sanjay Basu, MD PhD (Waymark)
Educational use - requires API keys and proper configuration for production use
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import openai
from bs4 import BeautifulSoup
import arxiv
import feedparser
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Research paper metadata"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    source: str  # 'pubmed', 'arxiv', 'journal', etc.
    url: str
    doi: Optional[str] = None
    journal: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    affected_chapters: List[int] = field(default_factory=list)
    paper_id: Optional[str] = None
    citation_count: int = 0
    impact_factor: Optional[float] = None

@dataclass
class MonitoringConfig:
    """Configuration for research monitoring"""
    # Data sources
    enable_pubmed: bool = True
    enable_arxiv: bool = True
    enable_journals: bool = True
    enable_conferences: bool = True
    
    # Search parameters
    lookback_days: int = 7
    max_papers_per_source: int = 50
    min_relevance_threshold: float = 0.6
    
    # API keys and endpoints
    pubmed_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Search terms for healthcare AI
    search_terms: List[str] = field(default_factory=lambda: [
        "machine learning healthcare",
        "artificial intelligence medicine",
        "clinical prediction models",
        "medical imaging AI",
        "natural language processing clinical",
        "reinforcement learning healthcare",
        "federated learning medical",
        "trustworthy AI healthcare",
        "bias fairness medical AI",
        "interpretable machine learning clinical",
        "generative AI healthcare",
        "large language models medicine",
        "multimodal AI medical",
        "AI deployment healthcare",
        "clinical decision support",
        "population health AI",
        "precision medicine machine learning",
        "digital health AI",
        "health equity AI",
        "regulatory AI healthcare"
    ])
    
    # Journal and conference sources
    journal_sources: List[str] = field(default_factory=lambda: [
        "nature_medicine",
        "nejm_ai", 
        "jama_network",
        "lancet_digital_health",
        "npj_digital_medicine",
        "jamia",
        "jmir",
        "ieee_tbme",
        "medical_image_analysis"
    ])
    
    conference_sources: List[str] = field(default_factory=lambda: [
        "icml",
        "neurips", 
        "iclr",
        "aaai",
        "ijcai",
        "amia",
        "himss",
        "miccai",
        "chil"
    ])

class PubMedMonitor:
    """Monitor PubMed for new healthcare AI research"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.rate_limit_delay = 0.34  # 3 requests per second max
        
    def search_papers(self, search_terms: List[str], days_back: int = 7, max_results: int = 50) -> List[ResearchPaper]:
        """Search PubMed for recent papers"""
        papers = []
        
        for term in search_terms:
            try:
                # Build search query
                query = self._build_search_query(term, days_back)
                
                # Search for paper IDs
                paper_ids = self._search_pubmed(query, max_results // len(search_terms))
                
                # Fetch paper details
                if paper_ids:
                    paper_details = self._fetch_paper_details(paper_ids)
                    papers.extend(paper_details)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error searching PubMed for term '{term}': {e}")
                continue
        
        # Remove duplicates
        papers = self._remove_duplicates(papers)
        
        logger.info(f"Found {len(papers)} papers from PubMed")
        return papers
    
    def _build_search_query(self, search_term: str, days_back: int) -> str:
        """Build PubMed search query"""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        date_filter = f"(\"{start_date.strftime('%Y/%m/%d')}\"[Date - Publication] : \"{end_date.strftime('%Y/%m/%d')}\"[Date - Publication])"
        
        # Combine search term with date filter
        query = f"({search_term}) AND {date_filter}"
        
        return query
    
    def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        """Search PubMed and return paper IDs"""
        search_url = f"{self.base_url}esearch.fcgi"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        paper_ids = data.get('esearchresult', {}).get('idlist', [])
        
        return paper_ids
    
    def _fetch_paper_details(self, paper_ids: List[str]) -> List[ResearchPaper]:
        """Fetch detailed information for papers"""
        if not paper_ids:
            return []
        
        fetch_url = f"{self.base_url}efetch.fcgi"
        
        params = {
            'db': 'pubmed',
            'id': ','.join(paper_ids),
            'retmode': 'xml'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        response = requests.get(fetch_url, params=params)
        response.raise_for_status()
        
        # Parse XML response
        papers = self._parse_pubmed_xml(response.text)
        
        return papers
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[ResearchPaper]:
        """Parse PubMed XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic information
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None and first_name is not None:
                            authors.append(f"{first_name.text} {last_name.text}")
                    
                    # Extract abstract
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    # Extract publication date
                    pub_date = self._extract_publication_date(article)
                    
                    # Extract DOI
                    doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
                    doi = doi_elem.text if doi_elem is not None else None
                    
                    # Extract journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else None
                    
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else None
                    
                    # Create paper object
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        publication_date=pub_date,
                        source='pubmed',
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                        doi=doi,
                        journal=journal,
                        paper_id=pmid
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error parsing PubMed article: {e}")
                    continue
        
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def _extract_publication_date(self, article) -> datetime:
        """Extract publication date from PubMed article"""
        # Try different date fields
        date_fields = [
            './/PubDate',
            './/ArticleDate',
            './/DateCompleted'
        ]
        
        for field in date_fields:
            date_elem = article.find(field)
            if date_elem is not None:
                year_elem = date_elem.find('Year')
                month_elem = date_elem.find('Month')
                day_elem = date_elem.find('Day')
                
                if year_elem is not None:
                    year = int(year_elem.text)
                    month = int(month_elem.text) if month_elem is not None else 1
                    day = int(day_elem.text) if day_elem is not None else 1
                    
                    try:
                        return datetime(year, month, day)
                    except ValueError:
                        continue
        
        # Default to current date if no valid date found
        return datetime.now()
    
    def _remove_duplicates(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity"""
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', paper.title.lower())
            title_hash = hashlib.md5(normalized_title.encode()).hexdigest()
            
            if title_hash not in seen_titles:
                seen_titles.add(title_hash)
                unique_papers.append(paper)
        
        return unique_papers

class ArxivMonitor:
    """Monitor arXiv for new healthcare AI research"""
    
    def __init__(self):
        self.client = arxiv.Client()
        
    def search_papers(self, search_terms: List[str], days_back: int = 7, max_results: int = 50) -> List[ResearchPaper]:
        """Search arXiv for recent papers"""
        papers = []
        
        # arXiv categories relevant to healthcare AI
        relevant_categories = [
            'cs.LG',  # Machine Learning
            'cs.AI',  # Artificial Intelligence
            'cs.CV',  # Computer Vision
            'cs.CL',  # Computation and Language
            'stat.ML',  # Machine Learning (Statistics)
            'q-bio.QM',  # Quantitative Methods
            'physics.med-ph'  # Medical Physics
        ]
        
        for term in search_terms:
            try:
                # Build search query
                query = self._build_arxiv_query(term, relevant_categories)
                
                # Search arXiv
                search = arxiv.Search(
                    query=query,
                    max_results=max_results // len(search_terms),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                # Process results
                for result in self.client.results(search):
                    # Check if paper is recent enough
                    if self._is_recent_paper(result.published, days_back):
                        paper = self._convert_arxiv_result(result)
                        papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error searching arXiv for term '{term}': {e}")
                continue
        
        # Remove duplicates
        papers = self._remove_duplicates(papers)
        
        logger.info(f"Found {len(papers)} papers from arXiv")
        return papers
    
    def _build_arxiv_query(self, search_term: str, categories: List[str]) -> str:
        """Build arXiv search query"""
        # Search in title, abstract, and categories
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Combine search term with categories
        query = f"({search_term}) AND ({category_query})"
        
        return query
    
    def _is_recent_paper(self, published_date: datetime, days_back: int) -> bool:
        """Check if paper is within the specified time range"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return published_date >= cutoff_date
    
    def _convert_arxiv_result(self, result) -> ResearchPaper:
        """Convert arXiv result to ResearchPaper object"""
        authors = [author.name for author in result.authors]
        
        paper = ResearchPaper(
            title=result.title,
            authors=authors,
            abstract=result.summary,
            publication_date=result.published,
            source='arxiv',
            url=result.entry_id,
            doi=result.doi,
            paper_id=result.entry_id.split('/')[-1],
            keywords=[category for category in result.categories]
        )
        
        return paper
    
    def _remove_duplicates(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers"""
        unique_papers = []
        seen_ids = set()
        
        for paper in papers:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                unique_papers.append(paper)
        
        return unique_papers

class JournalMonitor:
    """Monitor key journals for new healthcare AI research"""
    
    def __init__(self):
        self.journal_feeds = {
            'nature_medicine': 'https://www.nature.com/nm.rss',
            'nejm_ai': 'https://ai.nejm.org/rss/recent.xml',
            'jama_network': 'https://jamanetwork.com/rss/site_5/1145.xml',
            'lancet_digital_health': 'https://www.thelancet.com/journals/landig/rss.xml',
            'npj_digital_medicine': 'https://www.nature.com/npjdigitalmed.rss',
            'jamia': 'https://academic.oup.com/rss/site_5467/3113.xml',
            'jmir': 'https://www.jmir.org/rss/recent.xml'
        }
    
    def search_papers(self, journal_sources: List[str], days_back: int = 7) -> List[ResearchPaper]:
        """Search journal RSS feeds for recent papers"""
        papers = []
        
        for journal in journal_sources:
            if journal in self.journal_feeds:
                try:
                    feed_url = self.journal_feeds[journal]
                    journal_papers = self._parse_journal_feed(feed_url, journal, days_back)
                    papers.extend(journal_papers)
                    
                except Exception as e:
                    logger.error(f"Error monitoring journal '{journal}': {e}")
                    continue
        
        logger.info(f"Found {len(papers)} papers from journals")
        return papers
    
    def _parse_journal_feed(self, feed_url: str, journal_name: str, days_back: int) -> List[ResearchPaper]:
        """Parse RSS feed for a journal"""
        papers = []
        
        try:
            feed = feedparser.parse(feed_url)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries:
                # Parse publication date
                pub_date = self._parse_feed_date(entry)
                
                if pub_date >= cutoff_date:
                    # Extract paper information
                    paper = ResearchPaper(
                        title=entry.title,
                        authors=self._extract_authors(entry),
                        abstract=self._extract_abstract(entry),
                        publication_date=pub_date,
                        source=f'journal_{journal_name}',
                        url=entry.link,
                        journal=journal_name,
                        doi=self._extract_doi(entry)
                    )
                    
                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
        
        return papers
    
    def _parse_feed_date(self, entry) -> datetime:
        """Parse publication date from feed entry"""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        else:
            return datetime.now()
    
    def _extract_authors(self, entry) -> List[str]:
        """Extract authors from feed entry"""
        authors = []
        
        if hasattr(entry, 'authors'):
            for author in entry.authors:
                if hasattr(author, 'name'):
                    authors.append(author.name)
        elif hasattr(entry, 'author'):
            authors.append(entry.author)
        
        return authors
    
    def _extract_abstract(self, entry) -> str:
        """Extract abstract from feed entry"""
        if hasattr(entry, 'summary'):
            # Clean HTML tags
            soup = BeautifulSoup(entry.summary, 'html.parser')
            return soup.get_text()
        elif hasattr(entry, 'description'):
            soup = BeautifulSoup(entry.description, 'html.parser')
            return soup.get_text()
        else:
            return ""
    
    def _extract_doi(self, entry) -> Optional[str]:
        """Extract DOI from feed entry"""
        # Look for DOI in various fields
        text_fields = [
            getattr(entry, 'id', ''),
            getattr(entry, 'link', ''),
            getattr(entry, 'summary', '')
        ]
        
        doi_pattern = r'10\.\d{4,}\/[^\s]+'
        
        for text in text_fields:
            match = re.search(doi_pattern, text)
            if match:
                return match.group()
        
        return None

class RelevanceAnalyzer:
    """Analyze relevance of research papers to book chapters"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Chapter mapping for relevance analysis
        self.chapter_mapping = {
            1: ["clinical informatics", "healthcare data", "EHR", "medical records"],
            2: ["mathematical foundations", "statistics", "probability", "linear algebra"],
            3: ["data engineering", "FHIR", "DICOM", "medical imaging", "data pipelines"],
            4: ["structured machine learning", "clinical prediction", "feature engineering"],
            5: ["reinforcement learning", "treatment optimization", "dynamic treatment"],
            6: ["generative AI", "large language models", "text generation", "synthetic data"],
            7: ["AI agents", "autonomous systems", "multi-agent", "clinical workflows"],
            8: ["medical imaging", "computer vision", "radiology", "pathology", "imaging AI"],
            9: ["clinical NLP", "natural language processing", "clinical text", "medical language"],
            10: ["multimodal AI", "fusion", "imaging and text", "multimodal learning"],
            11: ["federated learning", "privacy", "distributed learning", "multi-institutional"],
            12: ["bias detection", "fairness", "algorithmic bias", "health equity"],
            13: ["interpretability", "explainable AI", "model interpretation", "clinical explanation"],
            14: ["robustness", "adversarial", "model reliability", "clinical safety"],
            15: ["uncertainty quantification", "calibration", "confidence intervals"],
            16: ["validation", "clinical trials", "prospective validation", "real-world evidence"],
            17: ["regulatory compliance", "FDA", "medical device", "clinical validation"],
            18: ["ethics", "responsible AI", "medical ethics", "AI governance"],
            19: ["deployment", "MLOps", "model deployment", "production systems"],
            20: ["monitoring", "drift detection", "model performance", "continuous learning"],
            21: ["integration", "EHR integration", "clinical workflows", "system integration"],
            22: ["scalability", "cloud computing", "distributed systems", "healthcare scale"],
            23: ["security", "privacy", "HIPAA", "data protection", "cybersecurity"],
            24: ["automation", "CI/CD", "automated deployment", "DevOps"],
            25: ["population health", "public health", "epidemiology", "health outcomes"],
            26: ["personalized medicine", "precision medicine", "individualized treatment"],
            27: ["future directions", "emerging technologies", "research frontiers"]
        }
    
    def analyze_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Analyze relevance of papers to book chapters"""
        analyzed_papers = []
        
        for paper in papers:
            try:
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(paper)
                
                # Identify affected chapters
                affected_chapters = self._identify_affected_chapters(paper)
                
                # Update paper with analysis results
                paper.relevance_score = relevance_score
                paper.affected_chapters = affected_chapters
                
                # Only include papers above threshold
                if relevance_score >= 0.6:  # Configurable threshold
                    analyzed_papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error analyzing paper '{paper.title}': {e}")
                continue
        
        # Sort by relevance score
        analyzed_papers.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Analyzed {len(papers)} papers, {len(analyzed_papers)} above relevance threshold")
        return analyzed_papers
    
    def _calculate_relevance_score(self, paper: ResearchPaper) -> float:
        """Calculate relevance score for a paper"""
        # Combine title and abstract for analysis
        text = f"{paper.title} {paper.abstract}"
        
        # Healthcare AI keywords with weights
        healthcare_keywords = {
            'machine learning': 0.8,
            'artificial intelligence': 0.8,
            'deep learning': 0.7,
            'neural network': 0.6,
            'clinical': 0.9,
            'medical': 0.9,
            'healthcare': 0.9,
            'hospital': 0.7,
            'patient': 0.8,
            'diagnosis': 0.8,
            'treatment': 0.8,
            'prediction': 0.7,
            'classification': 0.6,
            'regression': 0.5,
            'imaging': 0.7,
            'radiology': 0.7,
            'pathology': 0.7,
            'NLP': 0.6,
            'natural language': 0.6,
            'electronic health record': 0.8,
            'EHR': 0.8,
            'FHIR': 0.7,
            'federated learning': 0.8,
            'privacy': 0.6,
            'bias': 0.7,
            'fairness': 0.7,
            'interpretability': 0.7,
            'explainable': 0.7,
            'validation': 0.6,
            'deployment': 0.6,
            'population health': 0.8,
            'precision medicine': 0.8
        }
        
        # Calculate keyword-based score
        text_lower = text.lower()
        keyword_score = 0.0
        total_weight = 0.0
        
        for keyword, weight in healthcare_keywords.items():
            if keyword in text_lower:
                keyword_score += weight
                total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            keyword_score = keyword_score / total_weight
        
        # Boost score for high-impact journals
        journal_boost = 0.0
        if paper.journal:
            high_impact_journals = [
                'nature', 'science', 'cell', 'nejm', 'lancet', 'jama',
                'nature medicine', 'nature biotechnology'
            ]
            
            journal_lower = paper.journal.lower()
            for high_impact in high_impact_journals:
                if high_impact in journal_lower:
                    journal_boost = 0.2
                    break
        
        # Combine scores
        final_score = min(1.0, keyword_score + journal_boost)
        
        return final_score
    
    def _identify_affected_chapters(self, paper: ResearchPaper) -> List[int]:
        """Identify which chapters are affected by this paper"""
        affected_chapters = []
        text = f"{paper.title} {paper.abstract}".lower()
        
        for chapter_num, keywords in self.chapter_mapping.items():
            chapter_relevance = 0
            
            for keyword in keywords:
                if keyword.lower() in text:
                    chapter_relevance += 1
            
            # If multiple keywords match, consider chapter affected
            if chapter_relevance >= 2 or (chapter_relevance >= 1 and len(keywords) <= 3):
                affected_chapters.append(chapter_num)
        
        return affected_chapters
    
    def generate_ai_analysis(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Generate AI-powered analysis of paper relevance"""
        if not self.openai_api_key:
            return {}
        
        try:
            prompt = f"""
            Analyze the following research paper for its relevance to a healthcare AI textbook:
            
            Title: {paper.title}
            Abstract: {paper.abstract[:1000]}...
            
            Please provide:
            1. Relevance score (0-1) for healthcare AI applications
            2. Key contributions and innovations
            3. Potential impact on clinical practice
            4. Which chapters of a healthcare AI book this would be most relevant to
            5. Specific technical concepts introduced
            
            Respond in JSON format.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return {}

class ComprehensiveResearchMonitor:
    """Main class orchestrating comprehensive research monitoring"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Initialize monitors
        self.pubmed_monitor = PubMedMonitor(config.pubmed_api_key) if config.enable_pubmed else None
        self.arxiv_monitor = ArxivMonitor() if config.enable_arxiv else None
        self.journal_monitor = JournalMonitor() if config.enable_journals else None
        
        # Initialize analyzer
        self.relevance_analyzer = RelevanceAnalyzer(config.openai_api_key)
        
        logger.info("Comprehensive Research Monitor initialized")
    
    def monitor_all_sources(self) -> Dict[str, Any]:
        """Monitor all configured research sources"""
        all_papers = []
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'sources_monitored': [],
            'papers_found': 0,
            'relevant_papers': 0,
            'papers_by_source': {},
            'papers_by_chapter': {},
            'top_papers': [],
            'errors': []
        }
        
        # Monitor PubMed
        if self.pubmed_monitor:
            try:
                logger.info("Monitoring PubMed...")
                pubmed_papers = self.pubmed_monitor.search_papers(
                    self.config.search_terms,
                    self.config.lookback_days,
                    self.config.max_papers_per_source
                )
                all_papers.extend(pubmed_papers)
                monitoring_results['sources_monitored'].append('pubmed')
                monitoring_results['papers_by_source']['pubmed'] = len(pubmed_papers)
                
            except Exception as e:
                error_msg = f"PubMed monitoring error: {e}"
                logger.error(error_msg)
                monitoring_results['errors'].append(error_msg)
        
        # Monitor arXiv
        if self.arxiv_monitor:
            try:
                logger.info("Monitoring arXiv...")
                arxiv_papers = self.arxiv_monitor.search_papers(
                    self.config.search_terms,
                    self.config.lookback_days,
                    self.config.max_papers_per_source
                )
                all_papers.extend(arxiv_papers)
                monitoring_results['sources_monitored'].append('arxiv')
                monitoring_results['papers_by_source']['arxiv'] = len(arxiv_papers)
                
            except Exception as e:
                error_msg = f"arXiv monitoring error: {e}"
                logger.error(error_msg)
                monitoring_results['errors'].append(error_msg)
        
        # Monitor journals
        if self.journal_monitor:
            try:
                logger.info("Monitoring journals...")
                journal_papers = self.journal_monitor.search_papers(
                    self.config.journal_sources,
                    self.config.lookback_days
                )
                all_papers.extend(journal_papers)
                monitoring_results['sources_monitored'].append('journals')
                monitoring_results['papers_by_source']['journals'] = len(journal_papers)
                
            except Exception as e:
                error_msg = f"Journal monitoring error: {e}"
                logger.error(error_msg)
                monitoring_results['errors'].append(error_msg)
        
        # Analyze relevance
        logger.info("Analyzing paper relevance...")
        relevant_papers = self.relevance_analyzer.analyze_papers(all_papers)
        
        # Update results
        monitoring_results['papers_found'] = len(all_papers)
        monitoring_results['relevant_papers'] = len(relevant_papers)
        
        # Group papers by chapter
        for paper in relevant_papers:
            for chapter in paper.affected_chapters:
                if chapter not in monitoring_results['papers_by_chapter']:
                    monitoring_results['papers_by_chapter'][chapter] = []
                monitoring_results['papers_by_chapter'][chapter].append({
                    'title': paper.title,
                    'authors': paper.authors,
                    'source': paper.source,
                    'relevance_score': paper.relevance_score,
                    'url': paper.url
                })
        
        # Get top papers
        monitoring_results['top_papers'] = [
            {
                'title': paper.title,
                'authors': paper.authors,
                'source': paper.source,
                'relevance_score': paper.relevance_score,
                'affected_chapters': paper.affected_chapters,
                'url': paper.url,
                'abstract': paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            }
            for paper in relevant_papers[:10]  # Top 10 papers
        ]
        
        logger.info(f"Monitoring complete: {len(relevant_papers)} relevant papers found")
        
        return monitoring_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save monitoring results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Healthcare AI Research Monitor")
    parser.add_argument("--output-file", required=True, help="Output file for results")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back for papers")
    parser.add_argument("--max-papers", type=int, default=50, help="Maximum papers per source")
    parser.add_argument("--pubmed-api-key", help="PubMed API key")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MonitoringConfig(
        lookback_days=args.days_back,
        max_papers_per_source=args.max_papers,
        pubmed_api_key=args.pubmed_api_key,
        openai_api_key=args.openai_api_key
    )
    
    # Initialize monitor
    monitor = ComprehensiveResearchMonitor(config)
    
    # Run monitoring
    results = monitor.monitor_all_sources()
    
    # Save results
    monitor.save_results(results, args.output_file)
    
    # Print summary
    print(f"Monitoring complete!")
    print(f"Papers found: {results['papers_found']}")
    print(f"Relevant papers: {results['relevant_papers']}")
    print(f"Chapters affected: {len(results['papers_by_chapter'])}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
