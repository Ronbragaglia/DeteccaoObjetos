# Contribuindo para o Sistema de Detecção de Objetos

Obrigado por se interessar em contribuir com o Sistema de Detecção de Objetos! Este documento fornece diretrizes e instruções para contribuir com o projeto.

## 📋 Índice

- [Código de Conduta](#código-de-conduta)
- [Como Contribuir](#como-contribuir)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Padrões de Código](#padrões-de-código)
- [Testes](#testes)
- [Documentação](#documentação)
- [Relatando Issues](#relatando-issues)
- [Pull Requests](#pull-requests)

## 🤝 Código de Conduta

Ao participar deste projeto, você concorda em manter um ambiente respeitoso e inclusivo. Por favor:

- Seja respeitoso e construtivo
- Aceite feedback construtivo
- Foque no que é melhor para a comunidade
- Mostre empatia com outros contribuidores

## 🚀 Como Contribuir

### 1. Faça um Fork do Repositório

Clique no botão "Fork" no topo da página do repositório e clone seu fork:

```bash
git clone https://github.com/SEU_USUARIO/DeteccaoObjetos.git
cd DeteccaoObjetos
```

### 2. Crie uma Branch para sua Feature

```bash
git checkout -b feature/sua-feature
```

### 3. Faça suas Alterações

- Siga os padrões de código do projeto
- Adicione testes para novas funcionalidades
- Atualize a documentação conforme necessário

### 4. Commit suas Alterações

```bash
git add .
git commit -m "Add: sua mensagem de commit"
```

### 5. Push para sua Branch

```bash
git push origin feature/sua-feature
```

### 6. Abra um Pull Request

Vá para o repositório original e abra um Pull Request descrevendo suas alterações.

## 🔄 Processo de Desenvolvimento

### Configuração do Ambiente

1. Clone o repositório:

```bash
git clone https://github.com/Ronbragaglia/DeteccaoObjetos.git
cd DeteccaoObjetos
```

2. Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
# ou para desenvolvimento
pip install -e ".[dev]"
```

4. Configure as variáveis de ambiente:

```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

### Executando o Projeto

```bash
python main.py
```

### Executando os Testes

```bash
# Executar todos os testes
pytest

# Executar com cobertura
pytest --cov=src --cov-report=html

# Executar apenas testes de uma categoria
pytest -m unit
pytest -m integration
```

## 📐 Padrões de Código

### Formatação

Usamos [Black](https://black.readthedocs.io/) para formatação de código:

```bash
black src/ tests/
```

### Linting

Usamos [Flake8](https://flake8.pycqa.org/) para verificação de estilo:

```bash
flake8 src/ tests/
```

### Type Checking

Usamos [MyPy](https://mypy.readthedocs.io/) para verificação de tipos:

```bash
mypy src/
```

### Convenções de Nomenclatura

- **Classes**: `PascalCase` (ex: `ObjectDetector`)
- **Funções e Variáveis**: `snake_case` (ex: `detect_objects`)
- **Constantes**: `UPPER_SNAKE_CASE` (ex: `MAX_DETECTIONS`)
- **Módulos Privados**: `_snake_case` (ex: `_helper_function`)

### Padrões de Commit

Usamos o seguinte formato para mensagens de commit:

```
<Tipo>: <Descrição curta>

<Descrição detalhada opcional>

<Refs issues>
```

**Tipos permitidos:**

- `Add`: Adiciona nova funcionalidade
- `Fix`: Corrige um bug
- `Update`: Atualiza funcionalidade existente
- `Docs`: Adiciona ou atualiza documentação
- `Style`: Alterações de formatação/estilo
- `Refactor`: Refatoração de código
- `Test`: Adiciona ou atualiza testes
- `Chore`: Alterações de build/configuração

**Exemplos:**

```
Add: implementar detecção em tempo real

Implementa funcionalidade de detecção de objetos em tempo real usando webcam.

Closes #123
```

```
Fix: corrigir erro no sistema de áudio

Corrige bug que causava falha ao reproduzir áudio no Windows.

Fixes #456
```

## 🧪 Testes

### Escrevendo Testes

Todos os novos recursos devem incluir testes. Use o framework [pytest](https://docs.pytest.org/):

```python
import pytest
from src.detection.detector import ObjectDetector
from src.config.config import Config

def test_detector_initialization():
    """Testa inicialização do detector."""
    config = Config()
    detector = ObjectDetector(config)
    assert detector is not None
    assert detector.model is not None
```

### Marcadores de Teste

Use marcadores para categorizar testes:

```python
@pytest.mark.unit
def test_unit_example():
    """Teste unitário."""
    pass

@pytest.mark.integration
def test_integration_example():
    """Teste de integração."""
    pass

@pytest.mark.slow
def test_slow_example():
    """Teste lento."""
    pass
```

### Cobertura de Código

Mantenha a cobertura de código acima de 80%:

```bash
pytest --cov=src --cov-report=term-missing
```

## 📚 Documentação

### Docstrings

Use docstrings no formato Google:

```python
def detect_objects(self, frame: np.ndarray) -> List[Detection]:
    """
    Detecta objetos em um frame.
    
    Args:
        frame: Frame de entrada (imagem numpy array)
        
    Returns:
        List[Detection]: Lista de detecções encontradas
        
    Raises:
        RuntimeError: Se o modelo não foi carregado
        
    Example:
        >>> detector = ObjectDetector()
        >>> detections = detector.detect(frame)
    """
    pass
```

### Atualizando a Documentação

- Atualize o README.md quando adicionar novas funcionalidades
- Adicione exemplos em `examples/` para novos recursos
- Atualize o CHANGELOG.md para mudanças significativas

## 🐛 Relatando Issues

### Antes de Relatar

1. Verifique se o issue já foi reportado
2. Procure por issues similares
3. Verifique a documentação existente

### Criando um Issue

Ao criar um issue, inclua:

- **Título claro e descritivo**
- **Descrição detalhada do problema**
- **Passos para reproduzir**
- **Comportamento esperado vs. comportamento atual**
- **Ambiente** (SO, Python version, etc.)
- **Logs ou capturas de tela** (se aplicável)
- **Sugestões de solução** (opcional)

### Templates de Issue

Use os templates apropriados ao criar issues:

- **Bug Report**: Para relatar bugs
- **Feature Request**: Para solicitar novas funcionalidades
- **Documentation**: Para problemas na documentação
- **Question**: Para dúvidas e discussões

## 📥 Pull Requests

### Antes de Abrir um PR

1. Verifique se há issues relacionadas
2. Crie uma branch a partir de `main`
3. Siga os padrões de código
4. Adicione testes
5. Atualize a documentação
6. Certifique-se que todos os testes passam

### Criando um PR

Ao criar um Pull Request:

- **Título claro**: Descreva o que o PR faz
- **Descrição detalhada**: Explique as mudanças
- **Referências a issues**: Use `Fixes #123` ou `Closes #123`
- **Capturas de tela**: Para mudanças visuais
- **Checklist de revisão**: Marque itens completados

### Checklist de PR

- [ ] Código segue os padrões do projeto
- [ ] Testes adicionados/atualizados
- [ ] Documentação atualizada
- [ ] CHANGELOG.md atualizado
- [ ] Todos os testes passam
- [ ] Sem conflitos de merge
- [ ] Commits são claros e descritivos

### Processo de Revisão

1. O PR será revisado pelos mantenedores
2. Feedback será fornecido se necessário
3. Faça as alterações solicitadas
4. Aguarde aprovação
5. O PR será mergeado após aprovação

## 💡 Dicas de Contribuição

### Comece com Pequenas Contribuições

- Corrija typos na documentação
- Melhore exemplos
- Adicione testes
- Reporte bugs com detalhes

### Comunique-se

- Participe das discussões em issues
- Faça perguntas quando necessário
- Compartilhe suas ideias

### Aprenda com a Comunidade

- Leia PRs anteriores
- Estude o código existente
- Entenda a arquitetura do projeto

## 📞 Contato

Se você tiver dúvidas sobre como contribuir:

- Abra uma issue com a tag "question"
- Entre em contato com o mantenedor: ronbragaglia@gmail.com

## 📄 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a licença MIT do projeto.

---

**Obrigado por contribuir! 🎉**
