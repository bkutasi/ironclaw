<!-- Context: core/architecture/concepts | Priority: critical | Version: 1.0 | Updated: 2026-03-26 -->

# Document Extraction System

**Purpose**: Automated text extraction from document attachments in incoming messages, enabling LLM reasoning about file contents without direct file access.

**Last Updated**: 2026-03-26

## Quick Reference

**Update Triggers**: New format support | Size limit changes | Security modifications | Pipeline architecture changes

**Audience**: Developers, AI agents, security reviewers

**Key Files**:
- `src/document_extraction/mod.rs` — Middleware and processing pipeline
- `src/document_extraction/extractors.rs` — Format-specific extraction routines

**Security Model**: Inline data only (no URL downloading to prevent SSRF), size limits enforced, truncation for oversized content

---

## Architecture Overview

The document extraction system operates as middleware in the message processing pipeline. When a user sends a message with document attachments, the system automatically extracts text content before the LLM processes the message.

```
┌─────────────────────────────────────────────────────────────┐
│ Incoming Message with Attachments                           │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Message: "Analyze this document"                     │   │
│ │ Attachments:                                         │   │
│ │   - report.pdf (application/pdf)                     │   │
│ │   - data.xlsx (application/vnd.ms-excel)             │   │
│ └──────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ DocumentExtractionMiddleware                                │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│ │  Validate    │──│  Extract     │──│  Truncate    │      │
│ │  (size/kind) │  │  (by MIME)   │  │  (if needed) │      │
│ └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Processed Message                                           │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Message: "Analyze this document"                     │   │
│ │ Attachments:                                         │   │
│ │   - report.pdf                                       │   │
│ │     extracted_text: "Executive Summary: ..."         │   │
│ │   - data.xlsx                                        │   │
│ │     extracted_text: "Name\tAge\tRole\n..."           │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles**:
1. **Inline data only**: Channels must populate `attachment.data` before emitting messages
2. **No SSRF**: Downloading from `source_url` is intentionally not supported
3. **Graceful degradation**: Failed extraction produces user-friendly error messages
4. **Size enforcement**: 10 MB max document size, 100K char max extracted text

---

## DocumentExtractionMiddleware

**Location**: `src/document_extraction/mod.rs`

**Purpose**: Processes incoming messages, extracting text from document attachments based on MIME type.

### Processing Pipeline

The middleware executes the following steps for each attachment:

1. **Filter by Kind**: Skip non-document attachments (audio, video, images)
2. **Skip Already Extracted**: If `extracted_text` is already present, skip re-processing
3. **Size Validation**: Check against `MAX_DOCUMENT_SIZE` (10 MB)
4. **Data Availability**: Verify `attachment.data` is populated (inline data required)
5. **Format Detection**: Route to appropriate extractor based on MIME type
6. **Text Extraction**: Call format-specific extractor
7. **Truncation**: If extracted text exceeds `MAX_EXTRACTED_TEXT_LEN` (100K chars), truncate at char boundary
8. **Error Handling**: On failure, attach user-friendly error message

### Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_DOCUMENT_SIZE` | 10 MB (10 * 1024 * 1024 bytes) | Maximum attachment size to process |
| `MAX_EXTRACTED_TEXT_LEN` | 100,000 chars | Maximum extracted text length (~25K tokens) |

### Attachment Validation

**Size Checks** (performed at two points):

```rust
// Check 1: Using size_bytes metadata (if available)
if let Some(size) = attachment.size_bytes.filter(|&s| s > MAX_DOCUMENT_SIZE) {
    // Reject based on reported size
}

// Check 2: Using actual data length (definitive)
if attachment.data.len() as u64 > MAX_DOCUMENT_SIZE {
    // Reject based on actual data size
}
```

**Data Availability**:

```rust
// Inline data required — source_url downloading intentionally unsupported
if attachment.data.is_empty() {
    extractions.push((
        i,
        "[Document has no inline data. Please try sending the file again.]".to_string(),
    ));
    continue;
}
```

**Rationale**: Downloading from `source_url` would enable SSRF attacks. Channels must fetch and populate `attachment.data` via `store_attachment_data` before emitting the message.

### Error Messages

The middleware produces user-friendly error messages for common failure modes:

| Condition | Error Message |
|-----------|---------------|
| Document too large | `[Document too large for text extraction: {mb:.1} MB exceeds {max_mb:.0} MB limit. Please send a smaller file or copy-paste the relevant text.]` |
| No inline data | `[Document has no inline data. Please try sending the file again.]` |
| Extraction failed | `[Failed to extract text from '{name}' ({mime}): {error}. The file format may not be supported.]` |
| Text truncated | Appended: `[... truncated, document too long ...]` |

### Truncation Logic

To avoid panicking on multi-byte UTF-8 boundaries, truncation finds the last valid char boundary:

```rust
let text = if text.len() > MAX_EXTRACTED_TEXT_LEN {
    let boundary = text
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= MAX_EXTRACTED_TEXT_LEN)
        .last()
        .unwrap_or(0);
    let mut truncated = text[..boundary].to_string();
    truncated.push_str("\n\n[... truncated, document too long ...]");
    truncated
} else {
    text
};
```

---

## Extractors

**Location**: `src/document_extraction/extractors.rs`

**Purpose**: Format-specific text extraction routines, routed by MIME type.

### Supported Formats

| Category | MIME Types | Extensions | Extractor |
|----------|-----------|------------|-----------|
| **PDF** | `application/pdf` | `.pdf` | `extract_pdf()` |
| **Office XML** | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | `.docx` | `extract_docx()` |
| | `application/vnd.openxmlformats-officedocument.presentationml.presentation` | `.pptx` | `extract_pptx()` |
| | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` | `.xlsx` | `extract_xlsx()` |
| **Legacy Office** | `application/msword`, `application/vnd.ms-powerpoint`, `application/vnd.ms-excel` | `.doc`, `.ppt`, `.xls` | `extract_binary_strings()` |
| **Plain Text** | `text/plain`, `text/csv`, `text/markdown`, `text/html`, `text/xml`, `text/x-*` | `.txt`, `.csv`, `.md`, `.html`, `.xml` | `extract_utf8()` |
| **Code Files** | `text/x-python`, `text/x-java`, `text/x-rust`, `text/javascript`, etc. | `.py`, `.java`, `.rs`, `.js`, `.ts` | `extract_utf8()` |
| **Structured Data** | `application/json`, `application/xml`, `application/x-yaml`, `application/toml` | `.json`, `.xml`, `.yaml`, `.yml`, `.toml` | `extract_utf8()` |
| **RTF** | `application/rtf`, `text/rtf` | `.rtf` | `extract_rtf()` |

### Extraction Routing

```rust
pub fn extract_text(data: &[u8], mime: &str, filename: Option<&str>) -> Result<String, String> {
    let base_mime = mime.split(';').next().unwrap_or(mime).trim();

    match base_mime {
        "application/pdf" => extract_pdf(data),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => extract_docx(data),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => extract_pptx(data),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => extract_xlsx(data),
        
        // Legacy Office formats
        "application/msword" | "application/vnd.ms-powerpoint" | "application/vnd.ms-excel" => {
            extract_binary_strings(data)
        }
        
        // Plain text family
        "text/plain" | "text/csv" | "text/markdown" | "text/html" | ... => extract_utf8(data),
        
        // Structured data
        "application/json" | "application/xml" | "application/x-yaml" | ... => extract_utf8(data),
        
        // RTF
        "application/rtf" | "text/rtf" => extract_rtf(data),
        
        // Fallback: try filename extension
        _ => try_extract_by_extension(data, filename)
            .ok_or_else(|| format!("unsupported document type: {base_mime}")),
    }
}
```

---

## Format-Specific Extractors

### PDF Extraction

**Implementation**: Uses `pdf-extract` crate for text extraction.

```rust
fn extract_pdf(data: &[u8]) -> Result<String, String> {
    pdf_extract::extract_text_from_mem(data)
        .map(|t| t.trim().to_string())
        .map_err(|e| format!("PDF extraction failed: {e}"))
}
```

**Characteristics**:
- Extracts text content from PDF pages
- Trims leading/trailing whitespace
- May not preserve exact formatting or layout

### Office XML Extraction (DOCX, PPTX, XLSX)

**Common Pattern**: All Office XML formats are ZIP archives containing XML files.

**DOCX**:
```rust
fn extract_docx(data: &[u8]) -> Result<String, String> {
    extract_office_xml(data, "word/document.xml")
}
```

**PPTX** (PowerPoint):
- Extracts slide filenames from `ppt/slides/slide*.xml`
- Sorts slides in order
- Extracts text from each slide
- Joins with separator: `\n\n---\n\n`

**XLSX** (Excel):
- Reads shared strings from `xl/sharedStrings.xml`
- Reads sheet data from `xl/worksheets/sheet*.xml`
- Resolves shared string references (cell type `t="s"`)
- Outputs tab-separated rows, newline-separated sheets

**Common XML Extractor**:
```rust
fn extract_office_xml(data: &[u8], content_path: &str) -> Result<String, String> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("invalid Office XML archive: {e}"))?;

    let mut file = archive.by_name(content_path)
        .map_err(|e| format!("content file not found in archive: {e}"))?;

    let mut xml = String::new();
    file.read_to_string(&mut xml)
        .map_err(|e| format!("failed to read content: {e}"))?;

    let text = strip_xml_tags(&xml);
    if text.is_empty() {
        return Err("no text content found".to_string());
    }
    Ok(text)
}
```

### Plain Text Extraction

**Implementation**: UTF-8 decode with lossy fallback.

```rust
fn extract_utf8(data: &[u8]) -> Result<String, String> {
    match std::str::from_utf8(data) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Ok(String::from_utf8_lossy(data).to_string()),
    }
}
```

**Characteristics**:
- Attempts strict UTF-8 first
- Falls back to lossy decoding (replaces invalid sequences with )
- Preserves original content exactly

### RTF Extraction

**Implementation**: Basic control word stripping.

```rust
fn extract_rtf(data: &[u8]) -> Result<String, String> {
    let text = String::from_utf8_lossy(data);
    let mut result = String::new();
    let mut depth = 0i32;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => depth += 1,
            '}' => depth = (depth - 1).max(0),
            '\\' => {
                // Skip control word and numeric parameter
                // Convert common control words: \par → \n, \tab → \t
            }
            _ => {
                if depth <= 1 {
                    result.push(ch);
                }
            }
        }
    }
    
    let trimmed = result.trim().to_string();
    if trimmed.is_empty() {
        return Err("no text found in RTF".to_string());
    }
    Ok(trimmed)
}
```

**Supported Control Words**:
- `\par` / `\line` → newline
- `\tab` → tab

**Limitations**: Does not handle Unicode escapes (`\uN`), images, or complex formatting.

### Legacy Office Extraction (Binary Formats)

**Implementation**: Extract printable ASCII/UTF-8 runs from binary data.

```rust
fn extract_binary_strings(data: &[u8]) -> Result<String, String> {
    let mut strings = Vec::new();
    let mut current = String::new();

    for &byte in data {
        if (0x20..0x7F).contains(&byte) {
            current.push(byte as char);
        } else {
            if current.len() >= 4 {
                strings.push(std::mem::take(&mut current));
            }
            current.clear();
        }
    }
    if current.len() >= 4 {
        strings.push(current);
    }

    if strings.is_empty() {
        return Err("no readable text in binary document".to_string());
    }
    Ok(strings.join(" "))
}
```

**Characteristics**:
- Last-resort extraction for binary formats
- Extracts runs of 4+ printable ASCII characters
- May produce fragmented or incomplete text
- Does not preserve formatting or structure

### Filename Extension Fallback

When MIME type is generic or unrecognized, the system attempts extraction based on filename extension:

```rust
fn try_extract_by_extension(data: &[u8], filename: Option<&str>) -> Option<String> {
    let ext = filename?.rsplit('.').next()?.to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf(data).ok(),
        "docx" => extract_docx(data).ok(),
        "pptx" => extract_pptx(data).ok(),
        "xlsx" => extract_xlsx(data).ok(),
        "doc" | "ppt" | "xls" => extract_binary_strings(data).ok(),
        "rtf" => extract_rtf(data).ok(),
        "txt" | "csv" | "json" | "xml" | "yaml" | "md" | "py" | "js" | "ts" | "rs" | ... => {
            extract_utf8(data).ok()
        }
        _ => None,
    }
}
```

**Supported Extensions**: Over 30 code and text file extensions including `.gitignore`, `.dockerfile`, `.env`, `.log`, `.sql`, `.ini`, `.cfg`, `.conf`.

---

## XML Processing Utilities

### XML Tag Stripping

Used for DOCX, PPTX, XLSX content extraction:

```rust
fn strip_xml_tags(xml: &str) -> String {
    let mut result = String::with_capacity(xml.len() / 2);
    let mut in_tag = false;
    let mut last_was_space = true;

    for ch in xml.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                if !last_was_space && !result.is_empty() {
                    result.push(' ');
                    last_was_space = true;
                }
            }
            _ if !in_tag => {
                if ch.is_whitespace() {
                    if !last_was_space {
                        result.push(' ');
                        last_was_space = true;
                    }
                } else {
                    result.push(ch);
                    last_was_space = false;
                }
            }
            _ => {}
        }
    }

    // Decode common XML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .trim()
        .to_string()
}
```

**Features**:
- Collapses whitespace between text runs
- Decodes XML entities (`&amp;`, `&lt;`, etc.)
- Preserves text content only

### XLSX Shared Strings Parsing

```rust
fn parse_xlsx_shared_strings(xml: &str) -> Vec<String> {
    // Shared strings are in <si><t>text</t></si> elements
    let mut strings = Vec::new();
    let mut in_t = false;
    let mut current = String::new();

    for ch in xml.chars() {
        match ch {
            '<' => { /* track tag name */ }
            '>' => {
                if tag == "t" { in_t = true; current.clear(); }
                else if tag == "/t" { in_t = false; strings.push(current); }
            }
            _ if in_t => current.push(ch),
            _ => {}
        }
    }
    strings
}
```

### XLSX Sheet Parsing

```rust
fn parse_xlsx_sheet(xml: &str, shared_strings: &[String]) -> String {
    // Find <v> values in <c> cells
    // Resolve shared string refs (t="s" → index into shared_strings)
    // Output: tab-separated columns, newline-separated rows
}
```

---

## Document Types

### Classification by Processing Strategy

| Type | Processing | Complexity | Reliability |
|------|-----------|------------|-------------|
| **Native Text** | Direct UTF-8 decode | Low | High |
| **Structured Text** | UTF-8 decode (JSON/XML/YAML) | Low | High |
| **PDF** | `pdf-extract` crate | Medium | Medium-High |
| **Office XML** | ZIP + XML parsing | Medium-High | High |
| **RTF** | Control word stripping | Medium | Medium |
| **Legacy Binary** | Printable string extraction | Low | Low |

### MIME Type Mapping

**Text Family** (all use `extract_utf8`):
- `text/plain` — Plain text files
- `text/csv` — Comma-separated values
- `text/tab-separated-values` — TSV files
- `text/markdown` / `text/x-markdown` — Markdown documents
- `text/html` / `text/xhtml` — HTML documents
- `text/xml` — XML documents
- `text/x-python`, `text/x-java`, `text/x-rust`, etc. — Source code files
- `text/javascript`, `text/css` — Web languages
- `text/x-shellscript` — Shell scripts
- `text/x-toml`, `text/x-yaml` — Config formats
- `text/x-log` — Log files

**Application Types**:
- `application/pdf` — PDF documents
- `application/json` — JSON data
- `application/xml` — XML data
- `application/x-yaml` / `application/yaml` — YAML data
- `application/toml` — TOML data
- `application/rtf` / `text/rtf` — RTF documents
- `application/x-sh` — Shell scripts

**Office Formats**:
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document` — DOCX
- `application/vnd.openxmlformats-officedocument.presentationml.presentation` — PPTX
- `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` — XLSX
- `application/msword` — Legacy DOC
- `application/vnd.ms-powerpoint` — Legacy PPT
- `application/vnd.ms-excel` — Legacy XLS

---

## Processing Pipeline

### Message Flow

```
Channel (Telegram/Slack/etc.)
         │
         ▼
┌────────────────────────┐
│ emit_message()         │
│ - Populates            │
│   attachment.data      │
│ - Sets mime_type,      │
│   filename, size_bytes │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ DocumentExtraction     │
│ Middleware.process()   │
│ - Validates size/kind  │
│ - Routes to extractor  │
│ - Handles errors       │
│ - Truncates if needed  │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ Agent Loop             │
│ - Receives message     │
│   with extracted_text  │
│ - LLM can reason about │
│   document content     │
└────────────────────────┘
```

### Channel Responsibilities

Channels **must** populate attachment data before emitting messages:

```rust
// Channel implementation pattern
let mut attachment = IncomingAttachment {
    id: generate_id(),
    kind: AttachmentKind::Document,
    mime_type: "application/pdf".to_string(),
    filename: Some("report.pdf".to_string()),
    size_bytes: Some(file_size),
    source_url: None,  // Intentionally unused
    storage_key: None,
    extracted_text: None,
    data: file_bytes,  // MUST be populated
    duration_secs: None,
};

msg.attachments.push(attachment);
channel_host.emit_message(msg).await?;
```

**Critical**: `attachment.data` must contain the actual file bytes. The middleware does NOT download from `source_url`.

### Integration Point

The middleware is invoked from the agent's message processing pipeline:

```rust
// From agent loop (conceptual)
let doc_extractor = DocumentExtractionMiddleware::new();
doc_extractor.process(&mut incoming_message).await;

// Now message.attachments[i].extracted_text is populated
// Pass to LLM for reasoning
```

---

## Code Patterns

### Creating and Using the Middleware

```rust
use crate::document_extraction::DocumentExtractionMiddleware;

let middleware = DocumentExtractionMiddleware::new();
middleware.process(&mut incoming_message).await;

// Check extracted text
for attachment in &incoming_message.attachments {
    if let Some(text) = &attachment.extracted_text {
        println!("Extracted {} chars from {}", text.len(), attachment.filename.as_deref().unwrap_or("unknown"));
    }
}
```

### Direct Extractor Usage

```rust
use crate::document_extraction::extractors::extract_text;

// Extract from bytes with MIME type
let text = extract_text(&pdf_bytes, "application/pdf", Some("report.pdf"))?;

// Extract with filename fallback (generic MIME)
let text = extract_text(&data, "application/octet-stream", Some("notes.txt"))?;
```

### Testing Pattern

```rust
#[tokio::test]
async fn extracts_plain_text() {
    let middleware = DocumentExtractionMiddleware::new();
    let mut msg = IncomingMessage::new("test", "user1", "check this")
        .with_attachments(vec![
            doc_attachment("text/plain", "notes.txt", b"Hello world".to_vec()),
        ]);

    middleware.process(&mut msg).await;
    assert_eq!(
        msg.attachments[0].extracted_text.as_deref(),
        Some("Hello world")
    );
}

#[tokio::test]
async fn extracts_pdf_text() {
    let pdf_bytes = include_bytes!("../../tests/fixtures/hello.pdf");
    let middleware = DocumentExtractionMiddleware::new();
    let mut msg = IncomingMessage::new("test", "user1", "review")
        .with_attachments(vec![
            doc_attachment("application/pdf", "hello.pdf", pdf_bytes.to_vec()),
        ]);

    middleware.process(&mut msg).await;
    let text = msg.attachments[0].extracted_text.as_deref().unwrap_or("");
    assert!(text.contains("Hello"));
}
```

### Error Handling Pattern

```rust
match extractors::extract_text(&data, mime, filename) {
    Ok(text) => {
        // Truncate if needed
        let text = if text.len() > MAX_EXTRACTED_TEXT_LEN {
            // ... truncation logic ...
        } else {
            text
        };
        tracing::info!(
            attachment_id = %attachment.id,
            mime_type = %mime,
            text_len = text.len(),
            "Extracted text from document"
        );
        extractions.push((i, text));
    }
    Err(e) => {
        tracing::warn!(
            attachment_id = %attachment.id,
            mime_type = %mime,
            error = %e,
            "Failed to extract text from document"
        );
        let name = filename.unwrap_or("document");
        extractions.push((
            i,
            format!(
                "[Failed to extract text from '{name}' ({mime}): {e}. \
                 The file format may not be supported.]"
            ),
        ));
    }
}
```

---

## Security Considerations

### 1. SSRF Prevention

**Threat**: Malicious `source_url` pointing to internal network resources.

**Defense**: The middleware **only** processes `attachment.data` (inline bytes). Downloading from `source_url` is intentionally not implemented.

```rust
// Use inline data only — downloading from source_url is intentionally
// not supported to prevent SSRF. Channels must populate attachment.data
// via store_attachment_data before emitting the message.
if attachment.data.is_empty() {
    // Error: no inline data
}
```

**Channel Requirement**: Channels must fetch file content and populate `attachment.data` before calling `emit_message()`.

### 2. Size Limit Enforcement

**Threat**: Resource exhaustion via massive file uploads.

**Defense**: Two-tier size validation:

1. **Metadata check**: `attachment.size_bytes` (if available)
2. **Data check**: `attachment.data.len()` (definitive)

Both checked against `MAX_DOCUMENT_SIZE` (10 MB).

### 3. Truncation Safety

**Threat**: Panic on multi-byte UTF-8 boundary slicing.

**Defense**: Truncation finds last valid char boundary:

```rust
let boundary = text
    .char_indices()
    .map(|(i, _)| i)
    .take_while(|&i| i <= MAX_EXTRACTED_TEXT_LEN)
    .last()
    .unwrap_or(0);
let truncated = text[..boundary].to_string();
```

### 4. Format Validation

**Threat**: Malicious files masquerading as safe formats.

**Defense**: 
- Extractors are format-specific and defensive
- Binary string extraction is last-resort (low trust)
- Errors produce user-friendly messages, not stack traces

### 5. Memory Safety

**Threat**: Large file cloning causing memory spikes.

**Defense**: Size check before cloning:

```rust
// Enforce size limit before cloning to avoid unnecessary allocation
if attachment.data.len() as u64 > MAX_DOCUMENT_SIZE {
    // Reject early, no clone
}
let data = attachment.data.clone();  // Safe: already validated
```

---

## Related Files

**Core Implementation**:
- `src/document_extraction/mod.rs` — Middleware and processing pipeline (283 lines)
- `src/document_extraction/extractors.rs` — Format-specific extraction routines (515 lines)

**Integration Points**:
- `src/channels/` — Channel implementations (must populate `attachment.data`)
- `src/agent/` — Agent loop (invokes middleware)
- `src/channels/incoming_message.rs` — `IncomingMessage` and `IncomingAttachment` types

**Test Fixtures**:
- `tests/fixtures/hello.pdf` — Sample PDF for testing

**Dependencies**:
- `pdf-extract` — PDF text extraction
- `zip` — Office XML archive handling

---

## Related Context

- `channels-system.md` — Channel architecture and message emission
- `agent-system.md` — Agent loop and message processing
- `security-model.md` — Overall security architecture, SSRF prevention
- `architecture/concepts/config-precedence.md` — Configuration management

---

**Source Files**:
- `src/document_extraction/mod.rs` — Middleware implementation
- `src/document_extraction/extractors.rs` — Extractor routines
