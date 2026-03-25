# Makefile — TopoTS Čech core library
#
# Usage
# -----
#   make            # auto-detect platform, build shared library
#   make linux      # force Linux target
#   make macos      # force macOS target (universal binary x86_64 + arm64)
#   make windows    # cross-compile to Windows (requires mingw-w64)
#   make clean      # remove build artefacts
#   make test       # build and run the minimal smoke test
#
# Output: ../deps/lib/libcech.{so,dylib,dll}
# The Julia wrapper (CechFiltration.jl) looks for the library there.

CXX      ?= g++
SRC       = cech_core.cpp
OUTDIR    = ../deps/lib
INCFLAGS  = -I.
CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra $(INCFLAGS)

UNAME := $(shell uname -s 2>/dev/null || echo Windows)

.PHONY: all linux macos windows clean test

all:
	@mkdir -p $(OUTDIR)
ifeq ($(UNAME),Darwin)
	$(MAKE) macos
else ifeq ($(UNAME),Linux)
	$(MAKE) linux
else
	$(MAKE) windows
endif

linux:
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $(OUTDIR)/libcech.so $(SRC)
	@echo "Built $(OUTDIR)/libcech.so"

macos:
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC \
		-arch x86_64 -arch arm64 \
		-o $(OUTDIR)/libcech.dylib $(SRC) || \
	$(CXX) $(CXXFLAGS) -shared -fPIC \
		-o $(OUTDIR)/libcech.dylib $(SRC)
	@echo "Built $(OUTDIR)/libcech.dylib"

windows:
	@mkdir -p $(OUTDIR)
	$(CXX) $(CXXFLAGS) -shared -o $(OUTDIR)/cech.dll $(SRC) -Wl,--out-implib,$(OUTDIR)/cech.lib
	@echo "Built $(OUTDIR)/cech.dll"

clean:
	rm -f $(OUTDIR)/libcech.so $(OUTDIR)/libcech.dylib $(OUTDIR)/cech.dll \
	      $(OUTDIR)/cech.lib cech_test

test: $(SRC)
	$(CXX) $(CXXFLAGS) -DCECH_TEST_MAIN -o cech_test $(SRC) && ./cech_test
