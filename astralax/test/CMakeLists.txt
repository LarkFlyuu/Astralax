configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ASTL_OPT_TEST_DEPENDS
        FileCheck count not
        mlir-gen
        astl-opt
        astl-run
        )

add_lit_testsuite(check "Running the regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ASTL_OPT_TEST_DEPENDS}
        )
set_target_properties(check PROPERTIES FOLDER "Tests")

add_lit_testsuites(ASTL_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ASTL_OPT_TEST_DEPENDS})
