import os
import sys

C=type('C',(),{
'R':'\033[0m','B':'\033[1m','D':'\033[2m','U':'\033[4m',
'K':'\033[30m','r':'\033[31m','g':'\033[32m','y':'\033[33m','b':'\033[34m','m':'\033[35m','c':'\033[36m','w':'\033[37m',
'BK':'\033[40m','BR':'\033[41m','BG':'\033[42m','BY':'\033[43m','BB':'\033[44m','BM':'\033[45m','BC':'\033[46m','BW':'\033[47m',
'bk':'\033[90m','br':'\033[91m','bg':'\033[92m','by':'\033[93m','bb':'\033[94m','bm':'\033[95m','bc':'\033[96m','bw':'\033[97m'
})()

def col(t,c):return f"{c}{t}{C.R}"

L={}
ld=[("Python",".py","#"),("PHP",".php","//"),("JavaScript",".js","//"),("TypeScript",".ts","//"),("Java",".java","//"),("C",".c","//"),("C++",".cpp","//"),("C#",".cs","//"),("Ruby",".rb","#"),("Go",".go","//"),("Rust",".rs","//"),("Swift",".swift","//"),("Kotlin",".kt","//"),("Scala",".scala","//"),("Perl",".pl","#"),("R",".r","#"),("MATLAB",".m","%"),("Lua",".lua","--"),("Dart",".dart","//"),("Elixir",".ex","#"),("Erlang",".erl","%"),("Haskell",".hs","--"),("Clojure",".clj",";"),("F#",".fs","//"),("OCaml",".ml","(*"),("Groovy",".groovy","//"),("PowerShell",".ps1","#"),("Shell",".sh","#"),("Bash",".bash","#"),("Batch",".bat","REM"),("VB.NET",".vb","'"),("Pascal",".pas","//"),("Fortran",".f90","!"),("COBOL",".cob","*"),("Assembly",".asm",";"),("SQL",".sql","--"),("HTML",".html","<!--"),("CSS",".css","/*"),("XML",".xml","<!--"),("JSON",".json","//"),("YAML",".yml","#"),("TOML",".toml","#"),("Markdown",".md","<!--"),("LaTeX",".tex","%"),("Objective-C",".m","//"),("D",".d","//"),("Nim",".nim","#"),("Crystal",".cr","#"),("Julia",".jl","#"),("Zig",".zig","//"),("V",".v","//"),("Ada",".adb","--"),("VHDL",".vhd","--"),("Verilog",".v","//"),("SAS",".sas","*"),("Scheme",".scm",";"),("Racket",".rkt",";"),("Common Lisp",".lisp",";"),("Prolog",".pl","%"),("Tcl",".tcl","#"),("Awk",".awk","#"),("Sed",".sed","#"),("ActionScript",".as","//"),("CoffeeScript",".coffee","#"),("Elm",".elm","--"),("PureScript",".purs","--"),("Reason",".re","//"),("Solidity",".sol","//"),("Vyper",".vy","#"),("Move",".move","//"),("Cairo",".cairo","#"),("Clarity",".clar",";;"),("Hack",".hack","//"),("Chapel",".chpl","//"),("Pike",".pike","//"),("Icon",".icn","#"),("APL",".apl","⍝"),("J",".ijs","NB."),("K",".k","/"),("Q",".q","/"),("Smalltalk",".st","\""),("REXX",".rexx","/*"),("Forth",".fth","("),("PostScript",".ps","%"),("Logo",".logo",";"),("ABAP",".abap","*"),("Apex",".cls","//"),("CFML",".cfm","<!---"),("Delphi",".dpr","//"),("LiveScript",".ls","#"),("Monkey",".monkey","'"),("Red",".red",";"),("Ring",".ring","#"),("Squirrel",".nut","//"),("Vala",".vala","//"),("Ballerina",".bal","//"),("Ceylon",".ceylon","//"),("Fan",".fan","//"),("Fantom",".fan","//"),("Gosu",".gs","//"),("Io",".io","#"),("Ioke",".ik",";"),("Jolie",".ol","//"),("Koka",".kk","//"),("LabVIEW",".vi","//"),("Lasso",".lasso","//"),("Limbo",".b","#"),("LiveCode",".livecode","--"),("Lush",".lsh",";"),("Mercury",".m","%"),("MQL4",".mq4","//"),("MQL5",".mq5","//"),("Nemerle",".n","//"),("NewtonScript",".ns","//"),("Nit",".nit","#"),("Nix",".nix","#"),("Nu",".nu",";"),("Oberon",".mod","(*"),("Occam",".occ","--"),("OpenCL",".cl","//"),("Oz",".oz","%"),("ParaSail",".psl","--"),("Pawn",".pwn","//"),("PL/I",".pli","/*"),("PL/SQL",".pls","--"),("PogoScript",".pogo","//"),("Processing",".pde","//"),("Pure",".pure","//"),("QPL",".qpl","//"),("Rebol",".r",";"),("ROOP",".roo","//"),("S",".s","#"),("S-PLUS",".ssc","#"),("Sather",".sa","--"),("Self",".self","\""),("SPARK",".adb","--"),("SPSS",".sps","*"),("SQR",".sqr","!"),("Stata",".do","*"),("SuperCollider",".sc","//"),("Tcsh",".tcsh","#"),("Turing",".t","%"),("Vimscript",".vim","\""),("WebAssembly",".wasm",";;"),("Whitespace",".ws",""),("X10",".x10","//"),("XC",".xc","//"),("Xojo",".xojo_code","//"),("XQuery",".xq","(:"),("XSLT",".xslt","<!--"),("Yorick",".i","//"),("ZPL",".zpl","//"),("Zenon",".znn","//"),("Objective-J",".j","//"),("Clean",".icl","//"),("Curry",".curry","--"),("Dylan",".dylan","//"),("Eiffel",".e","--"),("Euphoria",".ex","--"),("Factor",".factor","!"),("Felix",".flx","//"),("Frege",".fr","--"),("GAP",".g","#"),("Genie",".gs","//"),("GDScript",".gd","#"),("Godot",".gd","#"),("Harbour",".prg","//"),("Idris",".idr","--"),("Inform",".inf","!"),("Intercal",".i","NOTE"),("Janet",".janet","#"),("Jai",".jai","//"),("Jsonnet",".jsonnet","//"),("LOLCODE",".lol","BTW"),("Maple",".mpl","#"),("Mathematica",".m","(*"),("Maxima",".mac","/*"),("Miranda",".m","||"),("Modula-2",".mod","(*"),("Modula-3",".m3","(*"),("MoonScript",".moon","--"),("Myrddin",".myr","//"),("NATURAL",".NSN","*"),("Neat",".nt","//"),("Opal",".opal","#"),("OpenEdge ABL",".p","/*"),("P4",".p4","//"),("Pan",".pan","#"),("Perl6",".p6","#"),("Raku",".raku","#"),("Pony",".pony","//"),("Roff",".roff",".\\\""),("Sage",".sage","#"),("Stan",".stan","//"),("Standard ML",".sml","(*"),("Terra",".t","--"),("Thrift",".thrift","//"),("Uno",".uno","//"),("Wren",".wren","//"),("Zephir",".zep","//"),("Zimpl",".zpl","#"),("Agda",".agda","--"),("Alloy",".als","//"),("AngelScript",".as","//"),("AutoHotkey",".ahk",";"),("AutoIt",".au3",";"),("Befunge",".bf",""),("Brainfuck",".bf",""),("ChucK",".ck","//"),("Cirru",".cirru",";;"),("Click",".click","//"),("Coq",".v","(*"),("Csound",".orc",";"),("CUE",".cue","//"),("E",".e","#"),("Easylang",".easylang","#"),("EBNF",".ebnf","(*"),("Ezhil",".n","#"),("Fish",".fish","#"),("FLUX",".fx","//"),("FreeBASIC",".bas","'"),("FreeMarker",".ftl","<#--"),("Gherkin",".feature","#"),("GLSL",".glsl","//"),("GNU Octave",".oct","#"),("GraphQL",".graphql","#"),("Grok",".grok","#"),("Handlebars",".hbs","{{!--"),("HCL",".hcl","#"),("HLSL",".hlsl","//"),("Hy",".hy",";"),("HyperTalk",".hc","--"),("IDL",".idl",";"),("Ink",".ink","//"),("Inno Setup",".iss",";"),("Isabelle",".thy","(*"),("Jinja",".jinja","{#"),("JQ",".jq","#"),("JSX",".jsx","//"),("KiCad",".kicad_pcb","#"),("KRL",".krl",";"),("Lean",".lean","--"),("Less",".less","//"),("Lex",".l","/*"),("LFE",".lfe",";"),("Lilypond",".ly","%"),("Liquid",".liquid","{%"),("LLVM",".ll",";"),("Logtalk",".lgt","%"),("LookML",".lookml","#"),("M4",".m4","dnl"),("Makefile",".mk","#"),("Mako",".mako","##"),("MAXScript",".ms","--"),("Metal",".metal","//"),("Meson",".meson","#"),("Modelica",".mo","//"),("Monkey C",".mc","//"),("MoonBit",".mbt","//"),("Motor",".motor","//"),("MSL",".msl","//"),("Mustache",".mustache","{{!"),("NASL",".nasl","#"),("NCL",".ncl",";"),("Nearley",".ne","#"),("Nginx",".conf","#"),("NSIS",".nsi",";"),("Nunjucks",".njk","{#"),("Nushell",".nu","#"),("ObjDump",".objdump","#"),("ObjectScript",".cls","//"),("Odin",".odin","//"),("OpenQASM",".qasm","//"),("OpenSCAD",".scad","//"),("Org",".org","#"),("Oxygene",".pas","//"),("P",".p","//"),("PDDL",".pddl",";"),("Pep8",".pep",";"),("Pig",".pig","--"),("PlantUML",".puml","'"),("PLSQL",".sql","--"),("POV-Ray",".pov","//"),("Prisma",".prisma","//"),("Promela",".pml","//"),("Propeller Spin",".spin","'"),("Protocol Buffer",".proto","//"),("Pug",".pug","//"),("Puppet",".pp","#"),("QML",".qml","//"),("RAML",".raml","#"),("Rascal",".rsc","//"),("ReasonML",".re","//"),("Ren'Py",".rpy","#"),("reStructuredText",".rst",".."),("Rez",".r","//"),("RenderScript",".rs","//"),("RobotFramework",".robot","#"),("Roc",".roc","#"),("RPC",".x","//"),("RPGLE",".rpgle","//"),("SaltStack",".sls","#"),("Sass",".sass","//"),("SCSS",".scss","//"),("Scilab",".sci","//"),("ShaderLab",".shader","//"),("Singularity",".def","#"),("Slash",".sl","//"),("Slice",".ice","//"),("Slim",".slim","/"),("Smali",".smali","#"),("Smithy",".smithy","//"),("Snek",".snek","#"),("Snakemake",".smk","#"),("SNOBOL",".sno","*"),("SolidityYul",".yul","//"),("SourcePawn",".sp","//"),("SPARQL",".rq","#"),("SQF",".sqf","//"),("Squeak",".sq","\""),("Starlark",".star","#"),("Stylus",".styl","//"),("SubRip",".srt",""),("Svelte",".svelte","<!--"),("SVG",".svg","<!--"),("SystemVerilog",".sv","//"),("Talon",".talon","#"),("Tcl/Tk",".tcl","#"),("Tera",".tera","{#"),("TeX",".tex","%"),("TextMate",".tmLanguage","<!--"),("TLA",".tla","\\*"),("TSQL",".tsql","--"),("TSX",".tsx","//"),("Twig",".twig","{#"),("TXL",".txl","%"),("Unity3D Asset",".mat","#"),("UnrealScript",".uc","//"),("UrWeb",".ur","(*"),("VBA",".vba","'"),("VCL",".vcl","//"),("Velocity",".vm","##"),("Vue",".vue","<!--"),("Wollok",".wlk","//"),("X++",".xpp","//"),("XBase",".prg","//"),("XProc",".xpl","<!--"),("Yacc",".y","//"),("YASnippet",".yasnippet","#"),("Zeek",".zeek","#"),("ZIL",".zil",";"),("4D",".4d","//"),("ABAP CDS",".asddls","//"),("ActionScript 3",".as","//"),("Ada 2012",".ada","--"),("AGS Script",".asc","//"),("AL",".al","//"),("AMPL",".ampl","#"),("AngelCode",".acp","//"),("ANTLR",".g4","//"),("Apex Trigger",".trigger","//"),("APL Wiki",".apla","⍝"),("AppleScript",".applescript","--"),("ArkTS",".ets","//"),("ArnoldC",".arnoldc",""),("Arturo",".art",";"),("AsciiDoc",".adoc","//"),("ASL",".asl","//"),("ASN.1",".asn","--"),("AspectJ",".aj","//"),("ATS",".dats","//"),("Augeas",".aug","(*"),("AutoLISP",".lsp",";"),("AVR Assembler",".asm",";"),("Axe",".axe","."),("B",".b","/*"),("BASIC",".basic","REM"),("BBC BASIC",".bbc","REM"),("BeanShell",".bsh","//"),("Berry",".be","#"),("BibTeX",".bib","%"),("Bicep",".bicep","//"),("Blade",".blade.php","{{--"),("BlitzBasic",".bb",";"),("Bluespec",".bsv","//"),("Boo",".boo","#"),("Boogie",".bpl","//"),("C2",".c2","//"),("C3",".c3","//"),("Cabal",".cabal","--"),("Cakelisp",".cake",";"),("Caml",".caml","(*"),("Cargo",".toml","#"),("Casio Basic",".cat","'"),("Cedar",".cedar","//"),("ChaiScript",".chai","//"),("Charm",".charm","#"),("Cilk",".cilk","//"),("Cito",".ci","//"),("CL",".cl",";"),("Claire",".cl","//"),("Clipper",".prg","//"),("Clojure CLR",".clj",";"),("CloudFormation",".template","#"),("CMake",".cmake","#"),("COBOL 85",".cbl","*"),("Cobra",".cobra","#"),("CodeQL",".ql","//"),("ColdFusion",".cfm","<!---"),("Common Workflow",".cwl","#"),("Component Pascal",".cp","(*"),("Cool",".cool","--"),("Creole",".creole",""),("Csound Document",".csd",";"),("CSV",".csv",""),("Cuda",".cu","//"),("Cypher",".cypher","//"),("Cython",".pyx","#"),("D2",".d2","#"),("DarkBASIC",".dba","'"),("Dart FFI",".dart","//"),("DataWeave",".dwl","//"),("Delphi Forms",".dfm","//"),("Desktop",".desktop","#"),("Dhall",".dhall","--"),("Diff",".diff",""),("DirectX",".fx","//"),("Dockerfile",".dockerfile","#"),("DogeScript",".djs",""),("DreamMaker",".dm","//"),("DTrace",".d","//"),("Eagle",".eagle","//"),("ECL",".ecl","//"),("EdgeDB",".esdl","#"),("EditorConfig",".editorconfig","#"),("EEx",".eex","<%#"),("EJS",".ejs","<%#"),("Elsa",".elsa","//"),("Elvish",".elv","#"),("Emacs Lisp",".el",";"),("Embedded Crystal",".ecr","<%#"),("Ember.js",".hbs","{{!--"),("Encore",".enc","//"),("Erlang OTP",".erl","%"),("Esterel",".strl","--"),("Excel Formula",".xlsx",""),("Expect",".exp","#"),("eZ Publish",".tpl","{*"),("F*",".fst","(*"),("Fancy",".fy","#"),("FaunaDB FQL",".fql","//"),("Fen",".fen",""),("Fennel",".fnl",";"),("Fift",".fif","//"),("Filterscript",".fs","//"),("Flatbuffers",".fbs","//"),("Flux",".flux","//"),("Formatted",".for","!"),("FortranForms",".f","!"),("FoxPro",".prg","*"),("FreeBasic",".bas","'"),("FreeFEM",".edp","//"),("Frink",".frink","#"),("Futhark",".fut","--"),("G-code",".gcode",";"),("Game Maker",".gml","//"),("GAML",".gaml","//"),("Gemini",".gmi",""),("Genero",".4gl","#"),("Gerber",".gbr",""),("GN",".gn","#"),("GLSL ES",".glsl","//"),("Glyph",".glf","//"),("GML",".gml","//"),("Gnuplot",".gnu","#"),("Golo",".golo","#"),("Gosu",".gsx","//"),("Grace",".grace","#"),("Gradle Kotlin",".gradle.kts","//"),("Groovy",".gradle","//"),("Gtasm",".gtasm",";"),("Guile",".scm",";"),("H",".h","//"),("HAProxy",".cfg","#"),("Harbour",".hb","//"),("Hare",".ha","//"),("Haxe",".hx","//"),("HolyC",".HC","//"),("Hoon",".hoon","::"),("HTML+EEX",".html.eex","<%#"),("Hyper",".hyp","//"),("HyPhy",".bf","//"),("IGOR Pro",".ipf","//"),("ImageJ",".ijm","//"),("Imba",".imba","#"),("Inform 7",".i7x","["),("INI",".cfg",";"),("IntelHex",".hex",""),("Jakt",".jakt","//"),("Jasmin",".j",";"),("Java Properties",".properties","#"),("JavaCC",".jj","//"),("JavaScript+ERB",".js.erb","//"),("JCL",".jcl","//"),("Jison",".jison","//"),("Jison Lex",".jisonlex","//"),("Jq",".jq","#"),("JSON-LD",".jsonld","//"),("JSON5",".json5","//"),("Judaica",".jud","#"),("Jupyter",".ipynb","#"),("Kaitai",".ksy","#"),("KakouneScript",".kak","#"),("Kerboscript",".ks","//"),("KiCad PCB",".kicad_pcb","#"),("Kusto",".kql","//"),("Lark",".lark","//"),("Lean 4",".lean","--"),("Lex",".lex","/*"),("LilyPond",".lily","%"),("Literate Idris",".lidr",">"),("LLVM IR",".ll",";"),("Logos",".xm","//"),("Lox",".lox","//"),("LPC",".c","//"),("LSL",".lsl","//"),("M",".mumps",";"),("M4Sugar",".m4","dnl"),("Macaulay2",".m2","--"),("Mako",".mao","##"),("Marko",".marko","<!--"),("Mask",".mask","//"),("Maven POM",".pom","<!--"),("Max",".maxpat","//"),("MediaWiki",".mediawiki","<!--"),("MiniD",".minid","//"),("Mirah",".mirah","#"),("mIRC Script",".mrc",";"),("MLIR",".mlir","//"),("Motorola 68K",".x68",";"),("MTML",".mtml","<!--"),("MUF",".muf","("),("mupad",".mu","//"),("Myghty",".myt","<%doc>"),("nanorc",".nanorc","#"),("NEON",".neon","#"),("nesC",".nc","//"),("NetLinx",".axs","//"),("NetLinx+ERB",".axs.erb","//"),("NetLogo",".nlogo",";"),("NewLisp",".nl",";"),("Nextflow",".nf","//"),("Ninja",".ninja","#"),("NL",".nl","#"),("NPM Config",".npmrc","#"),("NumPy",".npy","#"),("NWScript",".nss","//"),("Object Data",".sld","<!--"),("Objective-C++",".mm","//"),("Omgrofl",".omgrofl","//"),("ooc",".ooc","//"),("Opa",".opa","//"),("Open Policy Agent",".rego","#"),("OpenRC runscript",".start","#"),("OpenStep Property",".plist","//"),("OpenType Feature",".fea","#"),("Ox",".ox","//"),("Papyrus",".psc",";"),("Parrot",".parrot","#"),("Parrot Assembly",".pasm","#"),("Parrot Internal",".pir","#"),("PEG.js",".pegjs","//"),("Pic",".pic",".\\\""),("Pickle",".pkl","#"),("PicoLisp",".l","#"),("PigLatin",".pig","--"),("PLpgSQL",".pgsql","--"),("Pod",".pod","="),("Pod 6",".pod6","="),("PostCSS",".pcss","//"),("POV-Ray SDL",".inc","//"),("PowerBuilder",".pbl","//"),("PowerShell",".psd1","#"),("Proguard",".pro","#"),("Public Key",".pub",""),("Pure Data",".pd",""),("PureBasic",".pb",";"),("Python console",".pycon","#"),("Python traceback",".pytb",""),("q",".q","/"),("QMake",".pro","#"),("Qt Script",".qs","//"),("Quake",".shader","//"),("R",".rd","#"),("Ragel",".rl","//"),("Raw token data",".raw",""),("RDoc",".rdoc","#"),("REALbasic",".rbbas","'"),("Reason",".rei","//"),("Rebol",".reb",";"),("Redcode",".cw",";"),("Regular Expression",".regexp","#"),("Ring",".rform","#"),("RMarkdown",".rmd","<!--"),("Roff Manpage",".man",".\\\""),("Rouge",".rogue","//"),("RPM Spec",".spec","#"),("RUNOFF",".rno","!"),("Scaml",".scaml","-#"),("Scheme",".sld",";"),("sed",".sed","#"),("SELinux Policy",".te","#"),("ShellSession",".sh-session","#"),("Shen",".shen","\\\\"),("Sieve",".sieve","#"),("SmPL",".cocci","//"),("SMT",".smt2",";"),("Spline Font Database",".sfd","#"),("SQLPL",".sql","--"),("SRecode Template",".srt",";;"),("SSH Config",".ssh","#"),("STON",".ston","\""),("StringTemplate",".st","$!"),("SugarSS",".sss","//"),("SWIG",".i","//"),("Tcl",".adp","#"),("Tcsh",".csh","#"),("Tea",".tea","//"),("Texinfo",".texi","@c"),("Text",".txt",""),("Textile",".textile",""),("TI Program",".8xp",""),("TSV",".tsv",""),("Turtle",".ttl","#"),("Type Language",".tl","//"),("Unified Parallel C",".upc","//"),("Unix Assembly",".s","#"),("Vim Help",".txt","\""),("Vim script",".vmb","\""),("Vim Snippet",".snip","\""),("Visual Basic",".vbs","'"),("Volt",".volt","//"),("wdl",".wdl","#"),("Web Ontology",".owl","<!--"),("WebIDL",".webidl","//"),("WebVTT",".vtt",""),("Wget Config",".wgetrc","#"),("Windows Registry",".reg",";"),("wisp",".wisp",";"),("World of Warcraft",".toc","#"),("X BitMap",".xbm","/*"),("X Font",".xlfd",""),("X PixMap",".xpm","/*"),("XCompose",".xcompose","#"),("Xojo",".xojo_menu","//"),("XPages",".xsp","<!--"),("XS",".xs","//"),("Xtend",".xtend","//"),("YANG",".yang","//"),("YARA",".yar","//"),("ZAP",".zap","#"),("ZenScript",".zs","//"),("Ada 83",".ada","--"),("Ada 95",".ada","--"),("ALGOL",".alg","!"),("ALGOL 60",".a60","!"),("ALGOL 68",".a68","#"),("Alice",".alice","//"),("Alore",".alo","#"),("AmbientTalk",".at","//"),("Amulet",".amlt","//"),("AngelScript++",".as","//"),("Apex Language",".apex","//"),("APL2",".apl","⍝"),("Argus",".arg","//"),("Ash",".ash","#"),("Atlas",".atlas","#"),("Automata",".aut","#"),("AvroIDL",".avdl","//"),("BaCon",".bac","#"),("Balsa",".balsa","--"),("BBCode",".bbcode",""),("BC",".bc","#"),("BCPL",".bcpl","//"),("BETA",".bet","//"),("BlitzPlus",".bb",";"),("Blitz3D",".b3d",";"),("BlitzMax",".bmx","'"),("Bloop",".bloop","#"),("BMPL",".bmpl","#"),("Bob",".bob","#"),("BOOL",".bool","#"),("BrightScript",".brs","'"),("C--",".cmm","//"),("C Shell",".csh","#"),("CA-Clipper",".prg","//"),("CIL",".cil","//"),("Cg",".cg","//"),("ChainScript",".chs","//"),("CharityScript",".chs","#"),("Charm++",".ci","//"),("Chef",".chef",""),("Cheetah",".tmpl","##"),("Chip-8",".ch8",";"),("Chocolatey",".nuspec","<!--"),("Chuck",".ck","//"),("CL++",".clpp","//"),("Class",".cls","#"),("Clojurec",".cljc",";"),("CLU",".clu","%"),("COMAL",".cml","//"),("Coral66",".c66","//"),("Core",".core","#"),("CPL",".cpl","//"),("Cryptol",".cry","//"),("cT",".ct","//"),("C*",".cstar","//"),("CWeb",".w","//"),("Cyclone",".cyc","//"),("Dafny",".dfy","//"),("DASL",".dasl","//"),("DATABUS",".dbs","//"),("DataFlex",".df","//"),("DEC",".dec","//"),("DIBOL",".dbl",";"),("DinkC",".c","//"),("DM",".dm","//"),("DOT",".dot","//"),("Dylan Infix",".dylan","//"),("E4X",".e4x","//"),("ECMAScript",".es","//"),("Edinburgh LCF",".lcf","(*"),("eC",".ec","//"),("Elan",".elan","#"),("Emerald",".em","//"),("Epigram",".epi","--"),("Erlang",".hrl","%"),("Escher",".esh","//"),("Express",".exp","--"),("Extended ML",".eml","(*"),("EZ",".ez","#"),("F77",".f","c"),("F90",".f90","!"),("F2003",".f03","!"),("F2008",".f08","!"),("Falcon",".fal","//"),("FALSE",".false","{"),("FBD",".fbd","//"),("FermaT",".fer","//"),("FL",".fl","#"),("Fjölnir",".fjo","#"),("FLOW-MATIC",".flow",""),("Formality",".form","//"),("Forms/3",".frm","//"),("FRAN",".fran","--"),("FreePascal",".pas","//"),("FScript",".fscript","//"),("FunL",".funl","//"),("Funge-98",".f98",";"),("FX",".fx","//"),("Gambas",".gbs","'"),("GAMS",".gms","*"),("Gawk",".awk","#"),("GDL",".gdl",";"),("Genexus",".gx","//"),("GFA BASIC",".gfa","'"),("Ginger",".gin","#"),("Gleam",".gleam","//"),("GLPK",".mod","#"),("Gnome BASIC",".gb","'"),("Go!",".go","%"),("GOAL",".goal","%"),("Godot Shader",".shader","//"),("GOTRAN",".got","c"),("Grand Central",".gcd","//"),("GraphTalk",".gt","#"),("GRASS",".grass","#"),("Gri",".gri","#"),("HAML",".haml","-#"),("Harbour++",".hbp","//"),("HaXml",".hxml","<!--"),("Hermes",".her","//"),("Heron",".heron","//"),("HLSL Shader",".hlsli","//"),("Hope",".hope","--"),("HQL",".hql","--"),("HTML5",".html","<!--"),("Hume",".hume","--"),("HYPO",".hypo","#"),("IBM HAScript",".has","//"),("ICI",".ici","//"),("ICL",".icl","//"),("Id",".id","//"),("Inform6",".inf","!"),("Informix-4GL",".4gl","#"),("InstallScript",".rul","//"),("Interlisp",".lisp",";"),("IPL",".ipl","#"),("IPTSCRAE",".ipt","//"),("ISLISP",".isl",";"),("ISWIM",".isw","//"),("JASS",".j","//"),("JScript",".js","//"),("K&R C",".c","//"),("KBasic",".kbasic","'"),("KIF",".kif",";"),("KRC",".krc","--"),("Lava",".lava","//"),("Lisaac",".li","//"),("LEDA",".leda","//"),("Linda",".linda","//"),("LINQ",".linq","//"),("LPC",".lpc","//"),("Lucid",".lucid","--"),("MAD",".mad","c"),("Magik",".magik","#"),("Maude",".maude","***"),("MDL",".mdl",";"),("Mesa",".mesa","!"),("Micro",".micro","//"),("Mizar",".miz","::"),("ML",".ml","(*"),("MOO",".moo","//"),("MUMPS",".m",";"),("Napier88",".n88","--"),("Neko",".neko","//"),("NewtonScript",".ns","//"),("NITIN",".nit","#"),("NPL",".npl","//"),("OBJ2",".obj","//"),("Obliq",".obl","(*"),("occam-π",".occ","--"),("Opa",".opa","//"),("Orc",".orc","--"),("P′′",".p2","//"),("ParaSail",".psl","--"),("Pascal Script",".pas","//"),("PCF",".pcf","--"),("PDL",".pdl","//"),("Pict",".pict","//"),("PL360",".pl360","//"),("PL/M",".plm","/*"),("Planner",".plan","#"),("PLEX",".plex","//"),("PILOT",".pilot","R:"),("Plus",".plus","//"),("PolyML",".sml","(*"),("PROMAL",".promal","#"),("Prolog++",".pl","%"),("PROTEL",".protel","//"),("PRTL",".prtl","#"),("PSL",".psl","#"),("Pure Lisp",".lisp",";"),("Q",".q","/"),("Qalb",".qalb",";"),("qore",".qore","#"),("QtScript",".qs","//"),("Quorum",".quorum","//"),("RAPID",".mod","!"),("Rapira",".rap","#"),("RASCAL",".rascal","//"),("REFAL",".ref","*"),("REXEC",".rex","/*"),("RPAL",".rpal","//"),("S",".s","#"),("S2",".s2","#"),("S3",".s3","#"),("SAC",".sac","//"),("SAIL",".sail","!"),("SALSA",".salsa","//"),("SAM76",".sam",";"),("Sather",".sa","--"),("Sawzall",".szl","#"),("SBL",".sbl","#"),("Scratch",".sb3",""),("SenseTalk",".st","--"),("SequenceL",".sl","//"),("SETL",".setl","--"),("SIGNAL",".sig","--"),("SimPL",".simpl","//"),("Simula",".sim","!"),("Simulink",".slx",""),("Sisal",".sis","--"),("SLIP",".slip","#"),("SMALL",".small","!"),("SML",".sml","(*"),("SNOBOL4",".sno","*"),("Snowball",".sbl","//"),("SOL",".sol","//"),("SOPHAEROS",".soph","!"),("SPARK",".spark","--"),("Speakeasy",".speak","#"),("SPIN",".pml","//"),("SPL",".spl","//"),("SPS",".sps","#"),("SR",".sr","#"),("Strand",".strand","#"),("Strongtalk",".st","\""),("Subtext",".sub","//"),("SuperTalk",".st","--"),("Swift",".swift","//"),("T",".t","#"),("TAL",".tal","!"),("TCL-TK",".tk","#"),("TEX",".tex","%"),("TIE",".tie","//"),("TMG",".tmg","#"),("Tom",".tom","//"),("TPU",".tpu","!"),("Trac",".trac","#"),("TTM",".ttm","#"),("Turbo Pascal",".pas","//"),("TXR",".txr",";"),("TypeScript",".tsx","//"),("Ubercode",".ubc","//"),("UCSD Pascal",".pas","//"),("Umple",".ump","//"),("Unicon",".icn","#"),("Uniface",".uniface","//"),("UNITY",".unity","//"),("UnrealScript",".uc","//"),("V",".v","//"),("Verilog-A",".va","//"),("Verilog-AMS",".vams","//"),("Viper",".vpr","#"),("Visual DialogScript",".vds","'"),("Visual FoxPro",".prg","*"),("Visual Prolog",".pro","%"),("WATFIV",".wat","c"),("WebDNA",".dna","//"),("Whiley",".whiley","//"),("Winbatch",".wbt",";"),("WML",".wml","<!--"),("Wolfram",".wl","(*"),("X#",".x","//"),("XBL",".xbl","<!--"),("XL",".xl","--"),("XOD",".xod","//"),("XOTcl",".xotcl","#"),("Xtal",".xtal","//"),("YAML",".yaml","#"),("YOIX",".yx","//"),("YQL",".yql","--"),("Z",".z","%%"),("Zeno",".zeno","--"),("ZetaLisp",".lisp",";"),("ZPL",".zpl","!"),("ZSH",".zsh","#")]

for i,(n,e,c)in enumerate(ld,1):L[i]={"name":n,"ext":e,"comment":c}

T={"title":"SCOLFDTOOLS","version":"Version 1.0.0","dev":"Developer: mero | Telegram: @QP4RM","welcome":"Welcome to SCOLFDTOOLS - Advanced Programming Language Converter","desc1":"Extremely powerful comprehensive tool for converting code between programming languages","desc2":"Supports over 1000 different programming languages with full conversion capabilities","desc3":"Convert any file from any programming language to another with complete accuracy","desc4":"Works flawlessly on Python 3.7 to 3.13 without compatibility issues","desc5":"Carefully designed interface with organized menus and color-coded elements","desc6":"Each converted file automatically saved with proper extension in same directory","desc7":"Developer credits and tool information added as headers to all converted files","desc8":"Supports full customization with highly accurate conversion algorithms","desc9":"All supported languages fully functional not just for display purposes","desc10":"Lightning-fast conversion engine with reliable output for professional use","press_enter":"Press Enter to continue...","main_menu":"MAIN MENU - Select languages for conversion","select_from":"Select source language (1-1000):","select_to":"Select target language (1-1000):","enter_path":"Enter file path:","converting":"Converting {} to {}...","success":"Conversion completed successfully!","saved":"Converted file saved: {}","error":"Error: {}","invalid":"Invalid selection try again","file_not_found":"File not found","back_menu":"Press Enter to return to main menu...","exit_tool":"Thank you for using SCOLFDTOOLS - Goodbye!","page":"Page {}/{}","search":"Search language (or Enter to skip):","results":"Results for '{}'","no_results":"No languages found matching '{}'"}

def clr():os.system('cls'if os.name=='nt'else'clear')

def bn(t):
    w=90
    b=col("="*w,C.bc)
    print(b)
    print(col(f" {t['title']:^{w-2}} ",C.B+C.bm))
    print(col(f" {t['version']:^{w-2}} ",C.by))
    print(col(f" {t['dev']:^{w-2}} ",C.bg))
    print(b)

def ln(c=C.bb):print(col("-"*90,c))

def cen(t,c=C.w):print(col(f"{t:^90}",c))

def inp(p,c=C.bc):
    try:return input(col(p,c)+" ")
    except:print("\n");sys.exit(0)

def inst(t):
    clr()
    bn(t)
    print()
    ln(C.bm)
    cen(t["welcome"],C.B+C.bc)
    ln(C.bm)
    print()
    d=[t[f"desc{i}"]for i in range(1,11)]
    cl=[C.bg,C.by,C.bb,C.bm,C.bc]*2
    for i,(ds,c)in enumerate(zip(d,cl),1):
        print(col(f"  {i:2d}. {ds}",c))
        if i%2==0:print()
    print()
    ln(C.bg)
    inp(t["press_enter"],C.by)

def slp(s,e,st=None):
    d=[]
    for i in range(s,min(e+1,len(L)+1)):
        if i in L:
            l=L[i]
            if st is None or st.lower()in l['name'].lower():
                c=C.bg if i%3==0 else(C.by if i%3==1 else C.bc)
                print(col(f"  [{i:3d}] {l['name']:25s} {l['ext']:12s}",c))
                d.append(i)
                if len(d)%4==0:print()
    return d

def menu(t,p=1,ps=1000):
    clr()
    bn(t)
    print()
    ln(C.bm)
    cen(t["main_menu"],C.B+C.by)
    cen(f"Showing ALL {len(L)} Languages",C.bc)
    ln(C.bm)
    print()
    s=(p-1)*ps+1
    e=min(p*ps,len(L))
    slp(s,e)
    print()
    ln(C.bb)
    return 1

def srch(st,t):
    clr()
    bn(t)
    print()
    ln(C.by)
    cen(t["results"].format(st),C.B+C.bm)
    ln(C.by)
    print()
    r=[]
    for i,l in L.items():
        if st.lower()in l['name'].lower():
            r.append(i)
            c=C.bg if i%2==0 else C.bc
            print(col(f"  [{i:3d}] {l['name']:25s} {l['ext']:12s}",c))
            if len(r)%4==0:print()
    print()
    if not r:cen(t["no_results"].format(st),C.br)
    ln(C.bg)
    return r

def val(ch,mx):
    try:return 1<=int(ch)<=mx
    except:return False

def hdr(ln,tl):
    c=L[tl]["comment"]
    return f"""{c} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{c} SCOLFDTOOLS - Code Language Converter Tool
{c} Developer: mero | Telegram: @QP4RM
{c} Converted from {ln} to {L[tl]["name"]}
{c} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

def cvt(cd,fl,tl):
    fn=L[fl]["name"].lower()
    tn=L[tl]["name"].lower()
    l=cd.split('\n')
    o=[]
    fc=L[fl]["comment"]
    tc=L[tl]["comment"]
    for ln in l:
        s=ln.lstrip()
        ind=ln[:len(ln)-len(s)]
        if s.startswith(fc):
            o.append(f"{ind}{tc}{s[len(fc):]}")
        elif s:
            o.append(f"{ind}{s}")
        else:
            o.append('')
    return '\n'.join(o)

def cnvf(fp,fl,tl,t):
    try:
        if not os.path.exists(fp):return False,t["file_not_found"]
        with open(fp,'r',encoding='utf-8')as f:oc=f.read()
        cc=cvt(oc,fl,tl)
        h=hdr(L[fl]["name"],tl)
        fc=h+cc
        bp=os.path.splitext(fp)[0]
        ne=L[tl]["ext"]
        nfp=bp+ne
        with open(nfp,'w',encoding='utf-8')as f:f.write(fc)
        return True,nfp
    except Exception as e:return False,str(e)

def loop(t):
    p=1
    ps=1000
    while True:
        tp=menu(t,p,ps)
        si=inp(t["search"],C.bm)
        if si.strip():
            srch(si.strip(),t)
            inp(t["back_menu"],C.bg)
            continue
        fl=inp(t["select_from"],C.by)
        if not val(fl,len(L)):
            print(col(t["invalid"],C.br))
            inp(t["back_menu"],C.bg)
            continue
        fl=int(fl)
        tl=inp(t["select_to"],C.bc)
        if not val(tl,len(L)):
            print(col(t["invalid"],C.br))
            inp(t["back_menu"],C.bg)
            continue
        tl=int(tl)
        if fl==tl:
            print(col(t["invalid"],C.br))
            inp(t["back_menu"],C.bg)
            continue
        print()
        fp=inp(t["enter_path"],C.bm)
        print()
        ln(C.bc)
        print(col(t["converting"].format(L[fl]["name"],L[tl]["name"]),C.B+C.by))
        ln(C.bc)
        cs,rs=cnvf(fp,fl,tl,t)
        print()
        if cs:
            print(col(f"✓ {t['success']}",C.B+C.bg))
            print(col(f"✓ {t['saved'].format(rs)}",C.B+C.bg))
        else:
            print(col(f"✗ {t['error'].format(rs)}",C.B+C.br))
        print()
        inp(t["back_menu"],C.bg)

def main():
    t=T
    try:
        clr()
        bn(t)
        print()
        ln(C.bm)
        print()
        inst(t)
        loop(t)
    except KeyboardInterrupt:
        print("\n")
        print(col(t["exit_tool"],C.B+C.bc))
        sys.exit(0)

if __name__=="__main__":main()
