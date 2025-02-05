import os
import unittest
from dotenv import load_dotenv
from ingest import main
from langchain_community.document_loaders import (
    DirectoryLoader, 
    UnstructuredPDFLoader, 
    TextLoader,
    UnstructuredMarkdownLoader
)

class TestIngest(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.doc_dir = os.getenv('DOC_DIR')

    def test_directory_exists(self):
        """Test if the document directory exists"""
        self.assertTrue(os.path.exists(self.doc_dir), f"Directory does not exist: {self.doc_dir}")

    def test_directory_has_files(self):
        """Test if the directory contains supported files"""
        supported_extensions = {'.txt', '.pdf', '.md'}
        files = []
        for root, _, filenames in os.walk(self.doc_dir):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))
        
        self.assertGreater(len(files), 0, f"No supported files found in {self.doc_dir}")
        print(f"Found files: {files}")

    def test_file_permissions(self):
        """Test if we have read permissions for the directory and files"""
        self.assertTrue(os.access(self.doc_dir, os.R_OK), f"Cannot read directory: {self.doc_dir}")
        
        for root, _, filenames in os.walk(self.doc_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                self.assertTrue(os.access(filepath, os.R_OK), f"Cannot read file: {filepath}")

    def test_loader_initialization(self):
        """Test if loaders can be initialized"""
        test_files = {
            'test.pdf': UnstructuredPDFLoader,
            'test.txt': TextLoader,
            'test.md': UnstructuredMarkdownLoader
        }
        
        for filename, loader_class in test_files.items():
            try:
                loader_class(os.path.join(self.doc_dir, filename))
            except Exception as e:
                self.fail(f"Failed to initialize {loader_class.__name__}: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)