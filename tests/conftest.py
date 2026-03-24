"""
Configuração do pytest para testes do Sistema de Detecção de Objetos.
"""

import pytest
import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Fixture que fornece o diretório de dados de teste.
    
    Returns:
        Path: Caminho para o diretório de dados de teste
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_output_dir():
    """
    Fixture que fornece o diretório de saída de teste.
    
    Returns:
        Path: Caminho para o diretório de saída de teste
    """
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_image(test_data_dir):
    """
    Fixture que fornece uma imagem de teste.
    
    Args:
        test_data_dir: Diretório de dados de teste
        
    Returns:
        numpy.ndarray: Imagem de teste
    """
    import numpy as np
    
    # Criar uma imagem de teste simples
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Adicionar alguns padrões
    image[100:200, 100:200] = [255, 255, 255]  # Quadrado branco
    image[300:400, 300:400] = [255, 0, 0]       # Quadrado azul
    
    return image


@pytest.fixture
def sample_detections():
    """
    Fixture que fornece detecções de teste.
    
    Returns:
        list: Lista de detecções de teste
    """
    from src.detection.detector import Detection
    
    return [
        Detection(
            label="person",
            confidence=0.85,
            bbox=(10, 20, 100, 200)
        ),
        Detection(
            label="car",
            confidence=0.92,
            bbox=(50, 60, 150, 160)
        ),
        Detection(
            label="dog",
            confidence=0.78,
            bbox=(200, 250, 300, 350)
        ),
    ]


def pytest_configure(config):
    """
    Configuração adicional do pytest.
    
    Args:
        config: Configuração do pytest
    """
    # Adicionar marcadores personalizados
    config.addinivalue_line(
        "markers", "unit: marca testes como unitários"
    )
    config.addinivalue_line(
        "markers", "integration: marca testes como integração"
    )
    config.addinivalue_line(
        "markers", "slow: marca testes como lentos"
    )
