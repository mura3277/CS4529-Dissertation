package main

import (
	"fmt"
	"strings"

	"github.com/jonnenauha/obj-simplify/objectfile"
)

type Merge struct{}

type merger struct {
	Material string
	Objects  []*objectfile.Object
}

func (processor Merge) Name() string {
	return "Merge"
}

func (processor Merge) Desc() string {
	return "Merges objects and groups with the same material into a single mesh."
}

func (processor Merge) Execute(obj *objectfile.OBJ) error {
	// use an array to preserve original order and
	// to produce always the same output with same input.
	// Map will 'randomize' keys in golang on each run.
	materials := make([]*merger, 0)

	for _, child := range obj.Objects {
		// skip children that do not declare faces etc.
		if len(child.VertexData) == 0 {
			continue
		}
		found := false
		for _, m := range materials {
			if m.Material == child.Material {
				m.Objects = append(m.Objects, child)
				found = true
				break
			}
		}
		if !found {
			materials = append(materials, &merger{
				Material: child.Material,
				Objects:  []*objectfile.Object{child},
			})
		}
	}
	logInfo("  - Found %d unique materials", len(materials))

	mergeName := func(objects []*objectfile.Object) string {
		parts := []string{}
		for _, child := range objects {
			if len(child.Name) > 0 {
				parts = append(parts, child.Name)
			}
		}
		if len(parts) == 0 {
			parts = append(parts, "Unnamed")
		}
		name := strings.Join(parts, " ")
		// we might be merging hundreds or thousands of objects, at which point
		// the name becomes huge. Clamp with arbitrary 256 chars.
		if len(name) > 256 {
			name = ""
			for i, child := range objects {
				if len(child.Name) == 0 {
					continue
				}
				if len(name)+len(child.Name) < 256 {
					name += child.Name + " "
				} else {
					name += fmt.Sprintf("(and %d others)", len(objects)-i)
					break
				}
			}
		}
		return name
	}

	mergeComments := func(objects []*objectfile.Object) (comments []string) {
		for _, child := range objects {
			if len(child.Comments) > 0 {
				comments = append(comments, child.Comments...)
			}
		}
		return comments
	}

	// reset objects, we are about to rewrite them
	obj.Objects = make([]*objectfile.Object, 0)

	for _, merger := range materials {
		src := merger.Objects[0]
		child := obj.CreateObject(src.Type, mergeName(merger.Objects), merger.Material)
		child.Comments = mergeComments(merger.Objects)
		for _, original := range merger.Objects {
			child.VertexData = append(child.VertexData, original.VertexData...)
		}
	}

	return nil
}
