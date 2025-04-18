# AI Pattern Generator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> A tool for generating AI-driven patterns to integrate directly into Unity projects.

---

## 📖 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Executable](#building-the-executable)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Additional Restrictions](#additional-restrictions)
- [Attribution](#attribution)

---

## 🔍 Introduction

The **AI Pattern Generator** is a Python-based utility that leverages machine learning techniques to produce pattern data for use in Unity projects. Patterns are saved in a JSON-like format or can be directly embedded into Unity’s `StreamingAssets/Scripts` folder.

## ✨ Features

- Generate procedural patterns using AI models.
- Supports customizable parameters for pattern complexity and style.
- Seamless integration with Unity (drag & drop into `StreamingAssets`).
- Cross-platform: Windows, macOS, and Linux support via an executable.

## ⚙️ Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (recommended) or a compatible Python 3.8+ environment.
- System requirements: 4 GB RAM, multicore CPU.

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Pattern-Generator.git
   cd AI-Pattern-Generator
   ```

2. **Create and activate environment**
   ```bash
   conda env create -n AIPatternGenerator -f environment.yaml
   conda activate AIPatternGenerator
   ```

3. **Install dependencies (if not using Conda)**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Building the Executable

To package the tool into a stand-alone executable using PyInstaller:

1. Ensure you're in the project root:
   ```bash
   cd path/to/AI-Pattern-Generator
   ```
2. Clean previous builds:
   ```bash
   rm -rf build dist
   ```
3. Run PyInstaller with the provided spec:
   ```bash
   pyinstaller AI-PatternGenerator.spec
   ```
4. Once compilation completes, the executable will be available in the `dist/AI-Pattern-Generator` folder.

5. **Usage in Unity**: Drag the generated executable into your Unity project’s `StreamingAssets/Scripts` directory. Unity will load and execute it at runtime to produce pattern files.

---

## ▶️ Usage

Run the executable directly or call the Python script:

```bash
# Using the executable (after build)
./dist/AI-Pattern-Generator/AI-Pattern-Generator --config config.yml

# Or directly via Python
python main.py --config config.yml
```

Refer to `docs/USAGE.md` for detailed command-line options and examples.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourIdea`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature/YourIdea`).
5. Open a pull request.

Please ensure all new code is properly documented and tested.

---

## 📄 License

This project is licensed under the **Apache License, Version 2.0** (the "License"). You may not use this software except in compliance with the License. You may obtain a copy at:

> https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

---

## 🚫 Additional Restrictions

1. **Attribution**: Any derivative works, redistributions, or modifications must provide clear attribution to the Computer Science Department, Faculty of Science, Kasetsart University.
2. **Commercial Use**: Commercial use of this software or any derivative works requires explicit written permission from the Computer Science Department, Faculty of Science, Kasetsart University.

---

## 🎓 Attribution

Developed by the Computer Science Department, Faculty of Science, Kasetsart University.

---

*Happy pattern generating!*

