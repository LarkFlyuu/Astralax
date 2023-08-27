#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Astralax Helper

    Detects paths, libraries, executables, LLVM variables, etc.
"""

import os

from Logger import Logger


class ASTLHeader(object):
    """Detects paths, libraries, executables, LLVM variables, etc."""

    def __init__(self, loglevel):
        self.logger = Logger("astl.helper", loglevel)

    def findGitRoot(self, path):
        """Find the git root directory, if any, or return the input"""

        temp = path
        while temp:
            if os.path.exists(os.path.join(temp, ".git")):
                return temp
            temp = os.path.abspath(os.path.join(temp, os.pardir))
        return path

    def findASTLProgs(self, baseDir):
        """Find the necessary Astralax programs to run the benchmarks"""

        programs = {"astl-opt": "", "astl-run": ""}
        found = 0
        maxProgs = len(programs.keys())
        for root, dirs, files in os.walk(baseDir, followlinks=True):
            for prog in programs.keys():
                if prog in files:
                    programs[prog] = os.path.join(root, prog)
                    self.logger.debug(f"{prog}: {programs[prog]}")
                    found += 1
            if found == maxProgs:
                break

        if found < maxProgs:
            self.logger.error("Cannot find all Astralax programs")
            self.logger.error(f"Found: {programs}")
            return {}
        return programs
