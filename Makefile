#--------------------------------Master Makefile------------------------------#
# This is the master makefile. This makefile must be able to compile the
# project in the master branch of the repository.
#
# Copyright (C) 2010 by Allan Granados Jiménez (allangj1_618@hotmail.com)
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# The more important targets of this makefile are presented next
# - all     : Execute all the build process. I doesn't execute install or clean
# - build   : Build all the objects for the project
# - install : Install the binaries in a specific directory
# - clean   : Clean all the binaries created in the construction proccess
#
#-----------------------------------------------------------------------------#

SRCDIR		= src
BINDIR		= bin
INCLUDEDIR	= include
OBJECT		= facedetect
INSTALL_PREFIX	= 

.PHONY: all build install clean


all: build

build:
	@echo "Compiling the source code"
	${CXX} ${CFLAGS} ${LDFLAGS} -ggdb `pkg-config --cflags opencv` -o $(BINDIR)/$(OBJECT) $(SRCDIR)/$(OBJECT).cpp `pkg-config --libs opencv`

install:
	@echo "Installing binaries"
	@cp $(BINDIR)/* $(INSTALL_PREFIX)

clean:
	@echo "Removing the binaries"
	@rm -f $(BINDIR)/$(OBJECT)

