import os
import re
from typing import List

class RegexLineJoiner:
    def __init__(self):
        # Specific patterns for broken lines in legal documents
        self.broken_line_patterns = [
            # Pattern 1: Lines ending with "the" followed by line starting with lowercase
            (r'(\w+)\s+the\s*\n\s*([a-z])', r'\1 the \2'),
            
            # Pattern 2: Lines ending with lowercase followed by line starting with lowercase
            (r'([a-z])\s*\n\s*([a-z])', r'\1 \2'),
            
            # Pattern 3: Lines ending with comma followed by line starting with lowercase
            (r',\s*\n\s*([a-z])', r', \1'),
            
            # Pattern 4: Lines ending with word followed by line starting with lowercase
            (r'(\w)\s*\n\s*([a-z])', r'\1 \2'),
            
            # Pattern 5: Lines ending without sentence punctuation followed by line starting with lowercase
            (r'([^.!?])\s*\n\s*([a-z])', r'\1 \2'),
            
            # Pattern 6: Lines ending with "of" followed by line starting with lowercase
            (r'(\w+)\s+of\s*\n\s*([a-z])', r'\1 of \2'),
            
            # Pattern 7: Lines ending with "in" followed by line starting with lowercase
            (r'(\w+)\s+in\s*\n\s*([a-z])', r'\1 in \2'),
            
            # Pattern 8: Lines ending with "to" followed by line starting with lowercase
            (r'(\w+)\s+to\s*\n\s*([a-z])', r'\1 to \2'),
            
            # Pattern 9: Lines ending with "for" followed by line starting with lowercase
            (r'(\w+)\s+for\s*\n\s*([a-z])', r'\1 for \2'),
            
            # Pattern 10: Lines ending with "with" followed by line starting with lowercase
            (r'(\w+)\s+with\s*\n\s*([a-z])', r'\1 with \2'),
            
            # Pattern 11: Lines ending with "under" followed by line starting with lowercase
            (r'(\w+)\s+under\s*\n\s*([a-z])', r'\1 under \2'),
            
            # Pattern 12: Lines ending with "Section" followed by line starting with lowercase
            (r'(\w+)\s+Section\s*\n\s*([a-z])', r'\1 Section \2'),
            
            # Pattern 13: Lines ending with "Code" followed by line starting with lowercase
            (r'(\w+)\s+Code\s*\n\s*([a-z])', r'\1 Code \2'),
            
            # Pattern 14: Lines ending with "Act" followed by line starting with lowercase
            (r'(\w+)\s+Act\s*\n\s*([a-z])', r'\1 Act \2'),
            
            # Pattern 15: Lines ending with "Order" followed by line starting with lowercase
            (r'(\w+)\s+Order\s*\n\s*([a-z])', r'\1 Order \2'),
            
            # Pattern 16: Lines ending with "Court" followed by line starting with lowercase
            (r'(\w+)\s+Court\s*\n\s*([a-z])', r'\1 Court \2'),
            
            # Pattern 17: Lines ending with "Appellant" followed by line starting with lowercase
            (r'(\w+)\s+Appellant\s*\n\s*([a-z])', r'\1 Appellant \2'),
            
            # Pattern 18: Lines ending with "Respondent" followed by line starting with lowercase
            (r'(\w+)\s+Respondent\s*\n\s*([a-z])', r'\1 Respondent \2'),
            
            # Pattern 19: Lines ending with "Magistrate" followed by line starting with lowercase
            (r'(\w+)\s+Magistrate\s*\n\s*([a-z])', r'\1 Magistrate \2'),
            
            # Pattern 20: Lines ending with "Judge" followed by line starting with lowercase
            (r'(\w+)\s+Judge\s*\n\s*([a-z])', r'\1 Judge \2'),
        ]
        
        # Compile all patterns
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), replacement) 
                                 for pattern, replacement in self.broken_line_patterns]
    
    def join_broken_lines(self, text: str) -> str:
        """Join broken lines using regex patterns."""
        # Apply each pattern multiple times to catch all instances
        for _ in range(5):  # Multiple passes to catch all broken lines
            for pattern, replacement in self.compiled_patterns:
                text = pattern.sub(replacement, text)
        
        return text
    
    def process_file(self, input_file: str, output_file: str):
        """Process a single file to join broken lines."""
        print(f"Processing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into sections and process each section separately
        sections = re.split(r'(=== [A-Z]+ ===)', content)
        processed_sections = []
        
        for i, section in enumerate(sections):
            if section.startswith('=== ') and section.endswith(' ==='):
                # This is a section header, keep it as is
                processed_sections.append(section)
            else:
                # This is section content, process it
                if section.strip():
                    processed_content = self.join_broken_lines(section)
                    processed_sections.append(processed_content)
                else:
                    processed_sections.append(section)
        
        new_content = '\n'.join(processed_sections)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Processed {input_file}")
        return new_content
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all files in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        total_files = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)
                
                try:
                    self.process_file(input_file, output_file)
                    total_files += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        print(f"\nProcessing complete!")
        print(f"Total files processed: {total_files}")

def main():
    joiner = RegexLineJoiner()
    input_directory = 'Meta Splitted'
    output_directory = 'Broken Lines Joined'
    joiner.process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()


