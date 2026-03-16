# High-performance research environment
FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="Enzhe Lu"
LABEL research_area="Long-Context LLMs"

WORKDIR /research

# Install extra dependencies for attention research
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add custom Triton dependencies
RUN pip install --no-cache-dir triton==2.2.0

COPY . .

CMD ["/bin/bash"]
