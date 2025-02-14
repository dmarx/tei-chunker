# tei_chunker/service.py
"""
HTTP service for TEI document chunking.
"""
from typing import List, Dict, Any
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from loguru import logger

from .chunking import HierarchicalChunker

class ChunkingHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for document chunking."""
    
    chunker = HierarchicalChunker(
        max_chunk_size=20000,
        overlap_size=200
    )
    
    def do_POST(self) -> None:
        """Handle POST requests with XML content."""
        try:
            # Get content length
            content_length = int(self.headers['Content-Length'])
            
            # Read XML content
            xml_content = self.rfile.read(content_length).decode('utf-8')
            
            # Process the document
            sections = self.chunker.parse_grobid_xml(xml_content)
            chunks = self.chunker.chunk_document(sections)
            
            # Prepare response
            response = {
                'chunks': chunks,
                'chunk_count': len(chunks),
                'sections': [
                    {
                        'title': section.title,
                        'level': section.level,
                        'length': len(section.content),
                        'subsection_count': len(section.subsections)
                    }
                    for section in sections
                ]
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.send_error(500, str(e))
    
    def do_GET(self) -> None:
        """Handle GET requests with simple health check."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'status': 'healthy',
            'version': self.chunker.__version__
        }
        self.wfile.write(json.dumps(response).encode())


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the chunking service."""
    server = HTTPServer((host, port), ChunkingHandler)
    logger.info(f"Starting chunking service on {host}:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down chunking service")
        server.server_close()


if __name__ == "__main__":
    run_server()
