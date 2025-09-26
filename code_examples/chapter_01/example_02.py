"""
Chapter 1 - Example 2
Extracted from Healthcare AI Implementation Guide
"""

"""
Clinical Coding Systems Implementation for Healthcare AI

This module provides comprehensive support for working with clinical coding
systems (ICD-10, SNOMED CT, LOINC) in healthcare AI applications.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CodingSystem(Enum):
    """Enumeration of supported clinical coding systems."""
    ICD10_CM = "ICD-10-CM"
    ICD10_PCS = "ICD-10-PCS"
    SNOMED_CT = "SNOMED-CT"
    LOINC = "LOINC"
    CPT = "CPT"
    HCPCS = "HCPCS"

@dataclass
class ClinicalCode:
    """
    Represents a clinical code with full metadata for AI applications.
    
    This class provides comprehensive information about clinical codes
    including hierarchical relationships, semantic properties, and
    validation status that are essential for healthcare AI systems.
    """
    system: CodingSystem
    code: str
    display: str
    definition: Optional[str] = None
    parent_codes: List[str] = None
    child_codes: List[str] = None
    synonyms: List[str] = None
    is_active: bool = True
    effective_date: Optional[str] = None
    semantic_tags: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.parent_codes is None:
            self.parent_codes = []
        if self.child_codes is None:
            self.child_codes = []
        if self.synonyms is None:
            self.synonyms = []
        if self.semantic_tags is None:
            self.semantic_tags = []

class ICD10Processor:
    """
    Comprehensive ICD-10 code processor for healthcare AI applications.
    
    This class provides advanced ICD-10 code processing capabilities including
    validation, hierarchy navigation, semantic analysis, and clinical context
    interpretation that are essential for AI systems working with diagnostic data.
    """
    
    def __init__(self, enable_caching: bool = True):
        """
        Initialize ICD-10 processor with optional caching for performance.
        
        Args:
            enable_caching: Whether to cache code lookups for improved performance
        """
        self.enable_caching = enable_caching
        self.code_cache: Dict[str, ClinicalCode] = {}
        self.hierarchy_cache: Dict[str, List[str]] = {}
        
        \# ICD-10-CM code structure patterns
        self.icd10_cm_pattern = re.compile(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$')
        self.icd10_pcs_pattern = re.compile(r'^[0-9A-Z]{7}$')
        
        logger.info("ICD-10 processor initialized")
    
    def validate_icd10_code(self, code: str, system: CodingSystem = CodingSystem.ICD10_CM) -> Tuple[bool, str]:
        """
        Validate ICD-10 code format and structure.
        
        Args:
            code: ICD-10 code to validate
            system: ICD-10 coding system (CM or PCS)
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not code:
            return False, "Code cannot be empty"
        
        code = code.upper().strip()
        
        if system == CodingSystem.ICD10_CM:
            if not self.icd10_cm_pattern.match(code):
                return False, f"Invalid ICD-10-CM format: {code}"
            
            \# Additional validation rules for ICD-10-CM
            if len(code) < 3:
                return False, "ICD-10-CM codes must be at least 3 characters"
            
            \# Check for valid category (first 3 characters)
            category = code[:3]
            if not self._is_valid_icd10_category(category):
                return False, f"Invalid ICD-10-CM category: {category}"
                
        elif system == CodingSystem.ICD10_PCS:
            if not self.icd10_pcs_pattern.match(code):
                return False, f"Invalid ICD-10-PCS format: {code}"
        
        return True, "Valid ICD-10 code"
    
    def get_icd10_hierarchy(self, code: str) -> Dict[str, List[str]]:
        """
        Get hierarchical relationships for ICD-10 code.
        
        Args:
            code: ICD-10 code to analyze
            
        Returns:
            Dictionary containing parent and child codes
        """
        if self.enable_caching and code in self.hierarchy_cache:
            return self.hierarchy_cache[code]
        
        hierarchy = {
            'parents': self._get_parent_codes(code),
            'children': self._get_child_codes(code),
            'siblings': self._get_sibling_codes(code)
        }
        
        if self.enable_caching:
            self.hierarchy_cache[code] = hierarchy
        
        return hierarchy
    
    def analyze_diagnostic_patterns(self, diagnostic_codes: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns in diagnostic codes for AI applications.
        
        This method provides comprehensive analysis of diagnostic code patterns
        including comorbidity detection, disease category distribution, and
        clinical complexity assessment.
        
        Args:
            diagnostic_codes: List of ICD-10 diagnostic codes
            
        Returns:
            Comprehensive diagnostic pattern analysis
        """
        analysis = {
            'total_codes': len(diagnostic_codes),
            'unique_codes': len(set(diagnostic_codes)),
            'category_distribution': {},
            'chapter_distribution': {},
            'comorbidity_indicators': [],
            'complexity_score': 0.0,
            'chronic_conditions': [],
            'acute_conditions': [],
            'mental_health_indicators': [],
            'substance_use_indicators': []
        }
        
        \# Analyze each diagnostic code
        for code in diagnostic_codes:
            is_valid, _ = self.validate_icd10_code(code)
            if not is_valid:
                continue
            
            \# Extract category and chapter information
            category = code[:3]
            chapter = self._get_icd10_chapter(category)
            
            \# Update distributions
            analysis['category_distribution'][category] = analysis['category_distribution'].get(category, 0) + 1
            analysis['chapter_distribution'][chapter] = analysis['chapter_distribution'].get(chapter, 0) + 1
            
            \# Identify specific condition types
            if self._is_chronic_condition(code):
                analysis['chronic_conditions'].append(code)
            else:
                analysis['acute_conditions'].append(code)
            
            \# Identify mental health conditions
            if self._is_mental_health_condition(code):
                analysis['mental_health_indicators'].append(code)
            
            \# Identify substance use conditions
            if self._is_substance_use_condition(code):
                analysis['substance_use_indicators'].append(code)
        
        \# Calculate complexity score based on number of different chapters and chronic conditions
        analysis['complexity_score'] = self._calculate_diagnostic_complexity(analysis)
        
        \# Identify common comorbidity patterns
        analysis['comorbidity_indicators'] = self._identify_comorbidity_patterns(diagnostic_codes)
        
        return analysis
    
    def _is_valid_icd10_category(self, category: str) -> bool:
        """Validate ICD-10 category code."""
        \# Simplified validation - in production, would use official ICD-10 category list
        valid_ranges = [
            ('A00', 'B99'),  \# Infectious diseases
            ('C00', 'D49'),  \# Neoplasms
            ('D50', 'D89'),  \# Blood disorders
            ('E00', 'E89'),  \# Endocrine disorders
            ('F01', 'F99'),  \# Mental disorders
            ('G00', 'G99'),  \# Nervous system
            ('H00', 'H59'),  \# Eye disorders
            ('H60', 'H95'),  \# Ear disorders
            ('I00', 'I99'),  \# Circulatory system
            ('J00', 'J99'),  \# Respiratory system
            ('K00', 'K95'),  \# Digestive system
            ('L00', 'L99'),  \# Skin disorders
            ('M00', 'M99'),  \# Musculoskeletal
            ('N00', 'N99'),  \# Genitourinary
            ('O00', 'O9A'),  \# Pregnancy
            ('P00', 'P96'),  \# Perinatal
            ('Q00', 'Q99'),  \# Congenital
            ('R00', 'R99'),  \# Symptoms
            ('S00', 'T88'),  \# Injury
            ('V00', 'Y99'),  \# External causes
            ('Z00', 'Z99')   \# Health status
        ]
        
        for start, end in valid_ranges:
            if start <= category <= end:
                return True
        
        return False
    
    def _get_parent_codes(self, code: str) -> List[str]:
        """Get parent codes in ICD-10 hierarchy."""
        parents = []
        
        \# For ICD-10-CM, parents are less specific versions of the code
        if '.' in code:
            \# Remove the most specific part
            parent = code.rsplit('.', 1)<sup>0</sup>
            parents.append(parent)
            
            \# Continue up the hierarchy
            parents.extend(self._get_parent_codes(parent))
        elif len(code) > 3:
            \# Remove the last character for subcategory codes
            parent = code[:-1]
            parents.append(parent)
            parents.extend(self._get_parent_codes(parent))
        
        return parents
    
    def _get_child_codes(self, code: str) -> List[str]:
        """Get child codes in ICD-10 hierarchy."""
        \# In a production system, this would query an ICD-10 database
        \# For demonstration, return empty list
        return []
    
    def _get_sibling_codes(self, code: str) -> List[str]:
        """Get sibling codes (same parent) in ICD-10 hierarchy."""
        \# In a production system, this would query an ICD-10 database
        \# For demonstration, return empty list
        return []
    
    def _get_icd10_chapter(self, category: str) -> str:
        """Get ICD-10 chapter for a category code."""
        chapter_mapping = {
            ('A00', 'B99'): 'Infectious and parasitic diseases',
            ('C00', 'D49'): 'Neoplasms',
            ('D50', 'D89'): 'Diseases of blood and immune system',
            ('E00', 'E89'): 'Endocrine, nutritional and metabolic diseases',
            ('F01', 'F99'): 'Mental, behavioral and neurodevelopmental disorders',
            ('G00', 'G99'): 'Diseases of the nervous system',
            ('H00', 'H59'): 'Diseases of the eye and adnexa',
            ('H60', 'H95'): 'Diseases of the ear and mastoid process',
            ('I00', 'I99'): 'Diseases of the circulatory system',
            ('J00', 'J99'): 'Diseases of the respiratory system',
            ('K00', 'K95'): 'Diseases of the digestive system',
            ('L00', 'L99'): 'Diseases of the skin and subcutaneous tissue',
            ('M00', 'M99'): 'Diseases of the musculoskeletal system',
            ('N00', 'N99'): 'Diseases of the genitourinary system',
            ('O00', 'O9A'): 'Pregnancy, childbirth and the puerperium',
            ('P00', 'P96'): 'Perinatal conditions',
            ('Q00', 'Q99'): 'Congenital malformations',
            ('R00', 'R99'): 'Symptoms, signs and abnormal findings',
            ('S00', 'T88'): 'Injury, poisoning and external causes',
            ('V00', 'Y99'): 'External causes of morbidity',
            ('Z00', 'Z99'): 'Factors influencing health status'
        }
        
        for (start, end), chapter in chapter_mapping.items():
            if start <= category <= end:
                return chapter
        
        return 'Unknown chapter'
    
    def _is_chronic_condition(self, code: str) -> bool:
        """Determine if ICD-10 code represents a chronic condition."""
        \# Simplified chronic condition identification
        chronic_prefixes = [
            'E10', 'E11',  \# Diabetes
            'I10', 'I11', 'I12', 'I13',  \# Hypertension
            'I20', 'I21', 'I22', 'I25',  \# Coronary artery disease
            'J44', 'J45',  \# COPD, Asthma
            'N18',  \# Chronic kidney disease
            'F20', 'F31', 'F32', 'F33',  \# Mental health conditions
            'M05', 'M06',  \# Rheumatoid arthritis
            'K50', 'K51'   \# Inflammatory bowel disease
        ]
        
        return any(code.startswith(prefix) for prefix in chronic_prefixes)
    
    def _is_mental_health_condition(self, code: str) -> bool:
        """Determine if ICD-10 code represents a mental health condition."""
        return code.startswith('F')
    
    def _is_substance_use_condition(self, code: str) -> bool:
        """Determine if ICD-10 code represents a substance use condition."""
        substance_use_prefixes = ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19']
        return any(code.startswith(prefix) for prefix in substance_use_prefixes)
    
    def _calculate_diagnostic_complexity(self, analysis: Dict[str, Any]) -> float:
        """Calculate diagnostic complexity score."""
        \# Simplified complexity calculation
        chapter_count = len(analysis['chapter_distribution'])
        chronic_count = len(analysis['chronic_conditions'])
        mental_health_count = len(analysis['mental_health_indicators'])
        
        \# Base complexity on number of different body systems involved
        complexity = chapter_count * 0.3
        
        \# Add complexity for chronic conditions
        complexity += chronic_count * 0.2
        
        \# Add complexity for mental health comorbidities
        complexity += mental_health_count * 0.1
        
        return min(complexity, 10.0)  \# Cap at 10.0
    
    def _identify_comorbidity_patterns(self, codes: List[str]) -> List[str]:
        """Identify common comorbidity patterns."""
        patterns = []
        
        \# Check for diabetes + hypertension
        has_diabetes = any(code.startswith(('E10', 'E11')) for code in codes)
        has_hypertension = any(code.startswith(('I10', 'I11', 'I12', 'I13')) for code in codes)
        if has_diabetes and has_hypertension:
            patterns.append('Diabetes with hypertension')
        
        \# Check for COPD + heart disease
        has_copd = any(code.startswith('J44') for code in codes)
        has_heart_disease = any(code.startswith(('I20', 'I21', 'I22', 'I25')) for code in codes)
        if has_copd and has_heart_disease:
            patterns.append('COPD with cardiovascular disease')
        
        \# Check for mental health + substance use
        has_mental_health = any(code.startswith('F') and not code.startswith(('F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19')) for code in codes)
        has_substance_use = any(code.startswith(('F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19')) for code in codes)
        if has_mental_health and has_substance_use:
            patterns.append('Mental health with substance use disorder')
        
        return patterns


class SNOMEDCTProcessor:
    """
    SNOMED CT processor for advanced clinical terminology management.
    
    SNOMED CT provides the most comprehensive clinical terminology system,
    enabling sophisticated semantic analysis and clinical reasoning in AI applications.
    """
    
    def __init__(self, terminology_server_url: Optional[str] = None):
        """
        Initialize SNOMED CT processor.
        
        Args:
            terminology_server_url: URL for SNOMED CT terminology server
        """
        self.terminology_server_url = terminology_server_url or "https://snowstorm.ihtsdotools.org"
        self.concept_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("SNOMED CT processor initialized")
    
    def lookup_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up SNOMED CT concept by ID.
        
        Args:
            concept_id: SNOMED CT concept identifier
            
        Returns:
            Concept information including FSN, synonyms, and relationships
        """
        if concept_id in self.concept_cache:
            return self.concept_cache[concept_id]
        
        try:
            response = requests.get(
                f"{self.terminology_server_url}/MAIN/concepts/{concept_id}",
                timeout=10
            )
            response.raise_for_status()
            
            concept_data = response.json()
            
            \# Extract key information
            concept_info = {
                'concept_id': concept_id,
                'fsn': concept_data.get('fsn', {}).get('term'),
                'pt': concept_data.get('pt', {}).get('term'),
                'active': concept_data.get('active', False),
                'module_id': concept_data.get('moduleId'),
                'definition_status': concept_data.get('definitionStatus')
            }
            
            self.concept_cache[concept_id] = concept_info
            return concept_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to lookup SNOMED CT concept {concept_id}: {e}")
            return None
    
    def find_concepts_by_text(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find SNOMED CT concepts by text search.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching concepts
        """
        try:
            response = requests.get(
                f"{self.terminology_server_url}/MAIN/concepts",
                params={
                    'term': search_text,
                    'limit': limit,
                    'active': True
                },
                timeout=10
            )
            response.raise_for_status()
            
            search_results = response.json()
            concepts = []
            
            for item in search_results.get('items', []):
                concept = {
                    'concept_id': item.get('conceptId'),
                    'fsn': item.get('fsn', {}).get('term'),
                    'pt': item.get('pt', {}).get('term'),
                    'active': item.get('active', False)
                }
                concepts.append(concept)
            
            return concepts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search SNOMED CT concepts: {e}")
            return []


class LOINCProcessor:
    """
    LOINC processor for laboratory and clinical observation terminology.
    
    LOINC provides standardized codes for laboratory tests, clinical observations,
    and other measurements that are essential for AI applications processing
    clinical data.
    """
    
    def __init__(self):
        """Initialize LOINC processor."""
        self.loinc_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("LOINC processor initialized")
    
    def validate_loinc_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate LOINC code format.
        
        Args:
            code: LOINC code to validate
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not code:
            return False, "LOINC code cannot be empty"
        
        \# LOINC codes follow pattern: NNNNN-N (5 digits, hyphen, 1 check digit)
        loinc_pattern = re.compile(r'^\d{4,5}-\d$')
        
        if not loinc_pattern.match(code):
            return False, f"Invalid LOINC code format: {code}"
        
        \# Validate check digit (simplified - full implementation would use LOINC algorithm)
        return True, "Valid LOINC code format"
    
    def categorize_loinc_code(self, code: str) -> Dict[str, str]:
        """
        Categorize LOINC code by type and clinical domain.
        
        Args:
            code: LOINC code to categorize
            
        Returns:
            Dictionary with categorization information
        """
        \# Simplified categorization based on code ranges
        \# In production, would use official LOINC database
        
        code_num = int(code.split('-')<sup>0</sup>)
        
        if 1000 <= code_num <= 9999:
            return {
                'category': 'Laboratory',
                'subcategory': 'Chemistry',
                'domain': 'Clinical Laboratory'
            }
        elif 10000 <= code_num <= 19999:
            return {
                'category': 'Laboratory',
                'subcategory': 'Hematology',
                'domain': 'Clinical Laboratory'
            }
        elif 20000 <= code_num <= 29999:
            return {
                'category': 'Clinical',
                'subcategory': 'Vital Signs',
                'domain': 'Clinical Observation'
            }
        else:
            return {
                'category': 'Unknown',
                'subcategory': 'Unknown',
                'domain': 'Unknown'
            }


\# Demonstration and testing functions
def demonstrate_clinical_coding():
    """
    Demonstrate comprehensive clinical coding system usage.
    
    This function shows how to use the clinical coding processors
    for real-world healthcare AI applications.
    """
    print("=== Clinical Coding Systems Demonstration ===\n")
    
    \# Initialize processors
    icd10_processor = ICD10Processor(enable_caching=True)
    snomed_processor = SNOMEDCTProcessor()
    loinc_processor = LOINCProcessor()
    
    \# Example diagnostic codes for analysis
    sample_diagnostic_codes = [
        'E11.9',   \# Type 2 diabetes without complications
        'I10',     \# Essential hypertension
        'J44.1',   \# COPD with acute exacerbation
        'F32.9',   \# Major depressive disorder, single episode
        'N18.6',   \# End stage renal disease
        'Z51.11'   \# Encounter for chemotherapy
    ]
    
    print("1. ICD-10 Code Validation and Analysis")
    print("-" * 40)
    
    for code in sample_diagnostic_codes:
        is_valid, message = icd10_processor.validate_icd10_code(code)
        print(f"Code {code}: {'Valid' if is_valid else 'Invalid'} - {message}")
    
    \# Analyze diagnostic patterns
    print(f"\n2. Diagnostic Pattern Analysis")
    print("-" * 40)
    
    analysis = icd10_processor.analyze_diagnostic_patterns(sample_diagnostic_codes)
    print(f"Total codes analyzed: {analysis['total_codes']}")
    print(f"Unique codes: {analysis['unique_codes']}")
    print(f"Diagnostic complexity score: {analysis['complexity_score']:.2f}/10.0")
    print(f"Chronic conditions: {len(analysis['chronic_conditions'])}")
    print(f"Mental health indicators: {len(analysis['mental_health_indicators'])}")
    
    if analysis['comorbidity_indicators']:
        print("Identified comorbidity patterns:")
        for pattern in analysis['comorbidity_indicators']:
            print(f"  - {pattern}")
    
    print(f"\n3. Chapter Distribution")
    print("-" * 40)
    for chapter, count in analysis['chapter_distribution'].items():
        print(f"{chapter}: {count} codes")
    
    \# LOINC code validation
    print(f"\n4. LOINC Code Processing")
    print("-" * 40)
    
    sample_loinc_codes = ['33747-0', '2093-3', '8480-6', '8462-4']
    
    for loinc_code in sample_loinc_codes:
        is_valid, message = loinc_processor.validate_loinc_code(loinc_code)
        categorization = loinc_processor.categorize_loinc_code(loinc_code)
        
        print(f"LOINC {loinc_code}: {'Valid' if is_valid else 'Invalid'}")
        print(f"  Category: {categorization['category']}")
        print(f"  Subcategory: {categorization['subcategory']}")
        print(f"  Domain: {categorization['domain']}")
        print()


if __name__ == "__main__":
    demonstrate_clinical_coding()